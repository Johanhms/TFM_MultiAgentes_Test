import yfinance as yf
import pandas as pd
import time
import csv
import requests
import xml.etree.ElementTree as ET
import json
import pytz
from datetime import datetime, timedelta
from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas_ta as ta
import numpy as np
import joblib
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- CONFIGURACIÓN DE ACTIVOS (FLEXIBILIDAD MULTIACTIVO) ---
ASSET_MAPPING = {
    "BTC-USD": "BTCUSD", 
    "EURUSD=X": "EURUSD.sml", 
    "GBPUSD=X": "GBPUSD.sml", 
    "GC=F": "XAUUSD.sml",      
    "^GSPC": "US500",      
    "CL=F": "USOIL.sml",       
    "^DJI": "US30",            
    "NVDA": "NVDA_CFD.US"      
}

CURRENCY_MAP = {
    "BTC-USD": ["USD"], "EURUSD=X": ["EUR", "USD"], "GBPUSD=X": ["GBP", "USD"],
    "GC=F": ["USD"], "^GSPC": ["USD"], "CL=F": ["USD"], "^DJI": ["USD"], "NVDA": ["USD"]
}

# --- MATRIZ DE RIESGO Y DIRECCIÓN DE PORTAFOLIO ---
ASSET_VOLATILITY_WEIGHTS = {
    "BTC-USD": 0.10, "NVDA": 0.15, "GC=F": 0.20, "CL=F": 0.20,
    "^GSPC": 0.25, "^DJI": 0.25, "EURUSD=X": 0.30, "GBPUSD=X": 0.30 
}

ASSET_FUNDAMENTAL_SENSITIVITY = {
    "EURUSD=X": "HIGH", "GBPUSD=X": "HIGH", "GC=F": "HIGH", "CL=F": "HIGH",
    "^GSPC": "MEDIUM", "^DJI": "MEDIUM", "BTC-USD": "LOW", "NVDA": "LOW" 
}

MAX_GLOBAL_PORTFOLIO_RISK_PCT = 0.10  
MAX_PYRAMIDING_PER_ASSET = 3 # NUEVO: Límite de operaciones simultáneas por activo

# ==============================================================================
# FUNCIONES DE SOPORTE Y LOGGING
# ==============================================================================

def log_trade_execution(asset: str, ml_signal: str, ml_confidence: float, accion_final: str, ticket: int, precio_ejecucion: float, lot_size: float):
    """Genera la base de datos histórica (Backtest en vivo) para el futuro modelo de Meta-Labeling."""
    filename = "trading_history_log.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Asset', 'ML_Prediction', 'ML_Confidence', 'Executed_Action', 'MT5_Ticket', 'Execution_Price', 'Lot_Size'])
        
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), asset, ml_signal, round(ml_confidence, 4), accion_final, ticket, precio_ejecucion, lot_size])
    print(f"   📝 Registro guardado en {filename} para evaluación del Feedback Loop.")

def registrar_veto_csv(asset: str, ml_signal: str, motivo: str, precio: float, rsi: float, ema: float):
    """Guarda un registro tabular de cada vez que el Gestor de Portafolio bloquea a la IA."""
    archivo = "tfm_auditoria_vetos.csv"
    file_exists = os.path.isfile(archivo)
    
    with open(archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Fecha", "Activo", "Señal_ML_Bloqueada", "Motivo_Veto", "Precio_Evitado", "Nivel_RSI", "Nivel_EMA50"])
        
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([fecha_actual, asset, ml_signal, motivo, round(precio, 5), round(rsi, 2), round(ema, 5)])

def is_market_open(asset: str) -> bool:
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    
    if asset == "BTC-USD": return True
    if ny_time.weekday() == 5: return False
    if ny_time.weekday() == 6 and ny_time.hour < 17: return False
        
    if asset in ["NVDA", "^DJI", "^GSPC"]:
        if ny_time.hour < 9 or ny_time.hour >= 16: return False
        if ny_time.hour == 9 and ny_time.minute < 30: return False
        return True
        
    if asset in ["EURUSD=X", "GBPUSD=X", "GC=F", "CL=F"]:
        if ny_time.hour == 17: return False
        return True
    return True

def get_macro_calendar(currencies: list) -> list:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    eventos = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return []
        tree = ET.fromstring(response.content)
        for item in tree.findall('event'):
            country = item.find('country').text
            impact = item.find('impact').text
            if country in currencies and impact in ['High', 'Medium']:
                title = item.find('title').text
                date_str = item.find('date').text
                time_str = item.find('time').text
                eventos.append(f"- [{impact}] {country}: {title} ({date_str} {time_str})")
        return eventos[:5]
    except:
        return []

# ==============================================================================
# DEFINICIÓN DE ESTADO Y AGENTES (LANGGRAPH)
# ==============================================================================

class TradingState(TypedDict):
    asset: str
    current_price: float
    historical_data: pd.DataFrame
    technical_indicators: dict
    ml_prediction: str  
    ml_confidence: float
    fundamental_sentiment: str 
    final_signal: str          
    risk_params: dict   
    final_execution: str

# 1. Agente 1: Market Data Agent
def market_data_agent(state: TradingState):
    print(f"📡 [Market Data Agent] Obteniendo datos en vivo desde MT5 para {state['asset']}...")
    asset = state["asset"]
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    # Dependemos de la inicialización global en __main__
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 300)
    
    if velas_mt5 is None or len(velas_mt5) == 0:
        print(f"   ⚠️ No se pudieron obtener datos de MT5 para {symbol_mt5}.")
        return {"historical_data": pd.DataFrame(), "current_price": 0.0}
        
    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    current_price = float(df['Close'].iloc[-1])
    return {"historical_data": df, "current_price": current_price}

# 2. Agente 2: Technical Analyst
def technical_analyst_agent(state: TradingState):
    print("📊 [Technical Analyst] Ejecutando Feature Engineering y Análisis Direccional...")
    df = state["historical_data"].copy()
    
    # =====================================================================
    # 1. CÁLCULO CENTRALIZADO DE INDICADORES (Para TA y para ML)
    # =====================================================================
    # Osciladores y Momentum
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    roc = df.ta.roc(length=10)
    
    # Volatilidad
    atr = df.ta.atr(length=14)
    bbands = df.ta.bbands(length=20, std=2)
    
    # Tendencia y Régimen (Nuevas incorporaciones Quant)
    adx_df = df.ta.adx(length=14)
    df['EMA_20'] = df.ta.ema(length=20) # Tendencia Corta
    df['EMA_50'] = df.ta.ema(length=50) # Tendencia Media
    df['EMA_50_Slope'] = df['EMA_50'].diff(periods=3)
    
    # Flujo de Dinero Institucional (Volume Dimension)
    obv = df.ta.obv() 
    df['OBV'] = obv
    df['OBV_Slope'] = df['OBV'].diff(periods=3)
    
    # Features exclusivas para el modelo XGBoost (Centralizadas aquí)
    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1
        
    # Unimos todo el DataFrame limpio para que XGBoost lo consuma después
    df_completo = pd.concat([df, macd, rsi, atr, roc, adx_df], axis=1)
    
    # =====================================================================
    # 2. LÓGICA DE DECISIÓN DEL ANALISTA TÉCNICO (Confluencia Quant)
    # =====================================================================
    # Extraemos los valores exactos del momento presente (última vela cerrada)
    current_adx = float(adx_df['ADX_14'].iloc[-1]) if adx_df is not None else 0.0
    current_ema20 = float(df['EMA_20'].iloc[-1])
    current_ema50 = float(df['EMA_50'].iloc[-1])
    macd_hist = float(macd.iloc[-1, 2]) # Histograma del MACD
    
    tech_signal = "NEUTRAL"
    
    # Regla Institucional de Confluencia:
    # 1. ¿Hay fuerza en el mercado? (ADX > 20)
    # 2. ¿Quién domina la tendencia? (Cruce de EMAs)
    # 3. ¿El momentum apoya el movimiento? (MACD Histograma a favor)
    
    if current_adx > 20.0:
        if current_ema20 > current_ema50 and macd_hist > 0:
            tech_signal = "BULLISH"
            print("   📈 Análisis Técnico: Confluencia ALCISTA detectada (Golden Cross H1 + Momentum).")
        elif current_ema20 < current_ema50 and macd_hist < 0:
            tech_signal = "BEARISH"
            print("   📉 Análisis Técnico: Confluencia BAJISTA detectada (Death Cross H1 + Momentum).")
        else:
            print(f"   ⚖️ Análisis Técnico: Tendencia detectada pero sin momentum claro (ADX: {current_adx:.1f}).")
    else:
        print(f"   💤 Análisis Técnico: Régimen LATERAL (ADX: {current_adx:.1f} < 20). Riesgo de falsos rompimientos.")

    # Guardamos los indicadores clave para el Gestor de Portafolio
    technical_indicators = {
        "RSI_14": float(rsi.iloc[-1]),
        "MACD": float(macd.iloc[-1, 0]),          
        "MACD_Signal": float(macd.iloc[-1, 1]),   
        "ATR_14": float(atr.iloc[-1]),
        "ADX_14": current_adx, 
        "EMA_20": current_ema20,
        "EMA_50": current_ema50,
        "EMA_50_Slope": df['EMA_50_Slope'].iloc[-1],
        "BB_Upper": float(bbands.iloc[-1, 2]),    
        "BB_Lower": float(bbands.iloc[-1, 0]),
        "Tech_Signal": tech_signal # <--- El Agente ahora tiene opinión propia
    }
    
    # Actualizamos el estado. Sobrescribimos historical_data con el DF ya enriquecido.
    return {
        "technical_indicators": technical_indicators,
        "historical_data": df_completo 
    }

# 3. Agente 3: Quant ML Agent
def quant_ml_agent(state: TradingState):
    print("🧠 [Quant ML Agent] Consultando modelo de Inteligencia Artificial (Inferencia Rápida)...")
    
    asset = state["asset"]
    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    
    BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    model_path = BASE_DIR / f'quant_model_1h_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'
    
    # =====================================================================
    # 1. CARGA DEL MODELO Y AUDITORÍA DE SEGURIDAD
    # =====================================================================
    try:
        model = joblib.load(model_path)
        features_requeridas = joblib.load(features_path)
    except FileNotFoundError:
        print(f"   ❌ ERROR: Modelo descartado o no encontrado para {safe_name}. Forzando HOLD.")
        return {"ml_prediction": "HOLD", "ml_confidence": 0.0}
    
    # =====================================================================
    # 2. EXTRACCIÓN DIRECTA DE DATOS (Cero Latencia de Cálculo)
    # =====================================================================
    # El DataFrame ya viene enriquecido al 100% desde el Agente Técnico
    df_live = state["historical_data"]
    
    # Escudo protector: Verificamos que el Agente Técnico no haya olvidado ninguna columna 
    # que el modelo XGBoost necesite para pensar.
    columnas_faltantes = [f for f in features_requeridas if f not in df_live.columns]
    if columnas_faltantes:
        print(f"   ❌ ERROR CRÍTICO: Faltan variables en el dataset en vivo: {columnas_faltantes}")
        return {"ml_prediction": "HOLD", "ml_confidence": 0.0}
    
    # Aislamos exclusivamente la vela del momento presente (última fila)
    latest_data = df_live[features_requeridas].iloc[-1:]
    
    # Verificamos NaNs (Ocurre si MT5 acaba de abrir y las EMAs de 50 aún no tienen historial)
    if latest_data.isnull().values.any():
        print("   ⚠️ Datos incompletos (NaNs detectados en la última vela). Esperando maduración del mercado.")
        return {"ml_prediction": "HOLD", "ml_confidence": 0.0}
        
    # =====================================================================
    # 3. PREDICCIÓN PROBABILÍSTICA ESTOCÁSTICA
    # =====================================================================
    probabilities = model.predict_proba(latest_data)[0]
    prob_bajada = probabilities[0] # Probabilidad matemática de que caiga
    prob_subida = probabilities[1] # Probabilidad matemática de que suba
    
    # Umbral de disparo asimétrico
    UMBRAL = 0.52
    if prob_subida >= UMBRAL:
        prediction, confianza = "BUY", prob_subida
    elif prob_bajada > UMBRAL:
        prediction, confianza = "SELL", prob_bajada
    else:
        prediction, confianza = "HOLD", max(prob_subida, prob_bajada)
        
    print(f"   🤖 Predicción XGBoost: {prediction} (Confianza Matemática: {confianza*100:.1f}%)")
    
    return {"ml_prediction": prediction, "ml_confidence": confianza}

# 4. Agente 4: Fundamental Agent
def fundamental_analyst_agent(state: TradingState):
    asset = state["asset"]
    print(f"📰 [Fundamental Agent] Evaluando {asset}...")
    
    # =====================================================================
    # 1. SISTEMA DE CACHÉ INTRADÍA (TTL: Time-To-Live)
    # =====================================================================
    TTL_HOURS = 4  # El sentimiento expira y se recalcula cada 4 horas
    ahora = datetime.now()
    archivo_cache = "memoria_fundamental.json"
    memoria = {}
    
    if os.path.exists(archivo_cache):
        try:
            with open(archivo_cache, "r") as f: memoria = json.load(f)
        except Exception: pass
            
    # Verificar si el activo está en memoria y si la memoria aún es "fresca"
    if asset in memoria:
        datos_activo = memoria[asset]
        ultima_actualizacion_str = datos_activo.get("last_updated")
        
        if ultima_actualizacion_str:
            ultima_actualizacion = datetime.strptime(ultima_actualizacion_str, "%Y-%m-%d %H:%M:%S")
            diferencia_tiempo = ahora - ultima_actualizacion
            
            # Si la diferencia de tiempo es menor a 4 horas, usamos la caché para ahorrar Tokens
            if diferencia_tiempo < timedelta(hours=TTL_HOURS):
                sentimiento_guardado = datos_activo.get("sentiment", "NEUTRAL")
                tiempo_restante = TTL_HOURS - (diferencia_tiempo.total_seconds() / 3600)
                print(f"   ⚡ Usando memoria rápida intradía (Expira en {tiempo_restante:.1f}h): {sentimiento_guardado}")
                return {"fundamental_sentiment": sentimiento_guardado}

    # =====================================================================
    # 2. SI LA CACHÉ EXPIRÓ O NO EXISTE: Consultamos APIs y LLM
    # =====================================================================
    print(f"   📥 Caché expirada o vacía. Descargando datos macro frescos...")

    ticker = yf.Ticker(asset)
    news_list = ticker.news
    headlines = []
    if news_list:
        for n in news_list[:3]:
            if isinstance(n, dict):
                title = n.get('title') or (n.get('content', {}).get('title') if 'content' in n else None)
                if title: headlines.append(f"- {title}")
                
    news_text = "\n".join(headlines) if headlines else "Sin noticias."
    divisas = CURRENCY_MAP.get(asset, ["USD"])
    eventos_macro = get_macro_calendar(divisas)
    macro_text = "\n".join(eventos_macro) if eventos_macro else "Sin eventos macro."
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        prompt = f"""
        Eres un analista cuantitativo experto. Activo: {asset}.
        Noticias financieras: {news_text}
        Calendario macroeconómico: {macro_text}
        Sintetiza ambos datos. Responde ÚNICAMENTE con una de estas tres palabras: BULLISH, BEARISH o NEUTRAL.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        sentiment = response.content.strip().upper()
        if "BULLISH" in sentiment: sentiment = "BULLISH"
        elif "BEARISH" in sentiment: sentiment = "BEARISH"
        else: sentiment = "NEUTRAL"
    except Exception as e:
        print(f"   ❌ Error en LLM: {e}")
        sentiment = "NEUTRAL"
        
    print(f"   ✅ Nuevo sentimiento analizado: {sentiment}")
    
    # =====================================================================
    # 3. ACTUALIZAR CACHÉ CON EL NUEVO TIMESTAMP
    # =====================================================================
    memoria[asset] = {
        "sentiment": sentiment,
        "last_updated": ahora.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(archivo_cache, "w") as f:
        json.dump(memoria, f, indent=4)
        
    # Pausa de seguridad para no asfixiar el rate-limit de la API de Gemini/Yahoo
    time.sleep(3) 
    
    return {"fundamental_sentiment": sentiment}

# 5. Agente 5: Portfolio Manager Agent
def portfolio_manager_agent(state: TradingState):
    print("👔 [Portfolio Manager] Inciando Comité de Consenso y Auditoría de Riesgo...")
    
    asset = state.get("asset", "UNKNOWN")
    ml_signal = state.get("ml_prediction", "HOLD")
    ml_confidence = state.get("ml_confidence", 0.0)
    sentiment = state.get("fundamental_sentiment", "NEUTRAL")
    current_price = state.get("current_price")
    
    tech = state.get("technical_indicators", {})
    rsi = tech.get("RSI_14", 50.0)
    bb_upper = tech.get("BB_Upper", float('inf'))
    bb_lower = tech.get("BB_Lower", 0.0)
    adx = tech.get("ADX_14", 0.0)
    tech_signal = tech.get("Tech_Signal", "NEUTRAL") # La nueva opinión del Analista Técnico
    
    if ml_signal == "HOLD":
        return {"final_signal": "HOLD", "ml_confidence": ml_confidence}
        
    # =====================================================================
    # 1. VETO CUANTITATIVO: Régimen de Mercado (Filtro ADX)
    # =====================================================================
    if adx < 20:
        motivo = f"Régimen Lateral Detectado (ADX: {adx:.1f} < 20). Mercado sin tendencia."
        print(f"   🛑 VETO: {motivo}")
        registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, tech.get("EMA_50", 0))
        return {"final_signal": "HOLD", "ml_confidence": ml_confidence}

    # =====================================================================
    # 2. VETO TÉCNICO ASIMÉTRICO (Umbrales Dinámicos de Sobre-extensión)
    # =====================================================================
    # Cada activo tiene su propia "personalidad" de volatilidad
    RSI_THRESHOLDS = {
        "BTC-USD": {"overbought": 85, "oversold": 15}, # Cripto: Tolera extremos
        "NVDA": {"overbought": 82, "oversold": 18},    # Acciones Tech: Alta inercia
        "DEFAULT": {"overbought": 72, "oversold": 28}  # Forex/Índices: Revierten rápido
    }
    limites = RSI_THRESHOLDS.get(asset, RSI_THRESHOLDS["DEFAULT"])
    
    if ml_signal == "BUY":
        if rsi >= limites["overbought"] or current_price >= bb_upper:
            motivo = f"Sobrecompra Extrema para su clase (RSI: {rsi:.1f} > {limites['overbought']} o Toca BB)"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, tech.get("EMA_50", 0))
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}
            
    elif ml_signal == "SELL":
        if rsi <= limites["oversold"] or current_price <= bb_lower:
            motivo = f"Sobreventa Extrema para su clase (RSI: {rsi:.1f} < {limites['oversold']} o Toca BB)"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, tech.get("EMA_50", 0))
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}

    # =====================================================================
    # 3. COMITÉ DE CONSENSO (Machine Learning vs. Macro vs. Technical)
    # =====================================================================
    sensitivity = ASSET_FUNDAMENTAL_SENSITIVITY.get(asset, "HIGH")
    final_signal = ml_signal
    
    print(f"   ⚖️ Votación: ML=[{ml_signal}] | Técnico=[{tech_signal}] | Macro=[{sentiment}]")

    # Regla de Oro: Si la IA predice algo, pero AMBOS agentes humanos simulados 
    # (Técnico y Fundamental) opinan lo contrario, se tumba la operación.
    if ml_signal == "BUY" and tech_signal == "BEARISH" and sentiment == "BEARISH":
        print("   ⚠️ RECHAZO UNÁNIME: La Inteligencia Artificial alucina. Técnico y Macro son Bajistas. Abortando.")
        final_signal = "HOLD"
        
    elif ml_signal == "SELL" and tech_signal == "BULLISH" and sentiment == "BULLISH":
        print("   ⚠️ RECHAZO UNÁNIME: La Inteligencia Artificial alucina. Técnico y Macro son Alcistas. Abortando.")
        final_signal = "HOLD"
        
    else:
        # Si no hay rechazo unánime, aplicamos las reglas de sensibilidad del activo
        if sensitivity == "HIGH":
            # Para Forex y Oro: La macroeconomía manda. Un choque frontal aborta.
            if (ml_signal == "BUY" and sentiment == "BEARISH") or (ml_signal == "SELL" and sentiment == "BULLISH"):
                final_signal = "HOLD"
                print("   ⚠️ Conflicto Macro (Sensibilidad ALTA). Protegiendo capital.")
                
        elif sensitivity == "MEDIUM":
            # Para Índices (S&P500): Necesitamos que al menos la parte técnica valide a la IA si la Macro está en contra
            if (ml_signal == "BUY" and sentiment == "BEARISH" and tech_signal != "BULLISH"):
                final_signal = "HOLD"
                print("   ⚠️ Conflicto Mixto (Sensibilidad MEDIA). Sin apoyo técnico para contradecir Macro. Abortando.")
            elif (ml_signal == "SELL" and sentiment == "BULLISH" and tech_signal != "BEARISH"):
                final_signal = "HOLD"
                print("   ⚠️ Conflicto Mixto (Sensibilidad MEDIA). Sin apoyo técnico para contradecir Macro. Abortando.")
                
        elif sensitivity == "LOW":
            # Para Cripto/Acciones Tech: El modelo Quant es el rey, ignoramos ruido Macro.
            print(f"   🛡️ Sensibilidad Fundamental BAJA: Priorizando poder predictivo de IA ({ml_signal}).")

    if final_signal != "HOLD":
        print(f"   ✅ Ecosistema Alineado. Triangulación exitosa. Permiso de ejecución concedido.")
        
    print(f"   📋 Decisión Final: {final_signal} | Confianza Asociada: {ml_confidence*100:.1f}%")
    return {"final_signal": final_signal, "ml_confidence": ml_confidence}

# 6. Agente 6: Risk Manager Agent
def risk_manager_agent(state: TradingState):
    print("🛡️ [Risk Manager] Calculando dimensionamiento volumétrico y asimetría de riesgo...")
    
    final_signal = state.get("final_signal", "HOLD")
    asset = state.get("asset")
    confianza = state.get("ml_confidence", 0.0)
    tech = state.get("technical_indicators", {})
    atr = tech.get("ATR_14", 0.0)
    
    vacio = {"lot_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
    if final_signal == "HOLD" or atr <= 0: return {**vacio, "risk_params": vacio}
    
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    account_info = mt5.account_info()
    if account_info is None: return {**vacio, "risk_params": vacio}
    
    balance = account_info.balance 
    margen_libre = account_info.margin_free
    
    # =====================================================================
    # 1. ESCUDO DE LIQUIDEZ Y MARGEN INSTITUCIONAL
    # =====================================================================
    limite_liquidez = balance * 0.10
    if margen_libre < limite_liquidez:
        print(f"   🛑 RIESGO SISTÉMICO: Margen libre crítico (${margen_libre:,.2f}). Bloqueando nuevas operaciones.")
        return {**vacio, "risk_params": vacio}

    tick = mt5.symbol_info_tick(symbol_mt5)
    symbol_info = mt5.symbol_info(symbol_mt5)
    if tick is None or symbol_info is None: return {**vacio, "risk_params": vacio}

    # =====================================================================
    # 2. EVALUACIÓN DE CONFIANZA Y EXPECTATIVA DINÁMICA
    # =====================================================================
    # Ajustamos la agresividad volumétrica y el Take Profit según la IA
    if 0.50 <= confianza <= 0.529:
        multiplicador_lote, tp_mult, tier_text = 0.5, 1.5, "LOTE MÍNIMO (Scalp Corto)"
    elif 0.53 <= confianza <= 0.539:
        multiplicador_lote, tp_mult, tier_text = 1.0, 2.0, "LOTE MEDIO (Tendencia Estándar)"
    elif confianza >= 0.54:
        multiplicador_lote, tp_mult, tier_text = 1.5, 3.0, "LOTE MAYOR (Alta Convicción)"
    else:
        multiplicador_lote, tp_mult, tier_text = 0.5, 1.5, "LOTE DE MITIGACIÓN"

    sl_distance = atr * 1.5
    tp_distance = atr * tp_mult

    if final_signal == "BUY":
        precio_ejecucion_real = tick.ask
        sl = precio_ejecucion_real - sl_distance
        tp = precio_ejecucion_real + tp_distance
    else:
        precio_ejecucion_real = tick.bid
        sl = precio_ejecucion_real + sl_distance
        tp = precio_ejecucion_real - tp_distance

    volume_step = symbol_info.volume_step
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value 

    posiciones_activo = mt5.positions_get(symbol=symbol_mt5)
    num_posiciones = len(posiciones_activo) if posiciones_activo else 0
    
    # Bloqueo Estricto de Piramidación
    if num_posiciones >= MAX_PYRAMIDING_PER_ASSET:
        print(f"   🛑 LÍMITE DE RIESGO: Se alcanzó el máximo de {MAX_PYRAMIDING_PER_ASSET} operaciones en {asset}.")
        return {**vacio, "risk_params": vacio}
    
    # =====================================================================
    # 3. LÓGICA DE ESCALADO FRACCIONAL BASADO EN PnL (Piramidación Segura)
    # =====================================================================
    if num_posiciones > 0:
        ultima_posicion = sorted(posiciones_activo, key=lambda p: p.time)[-1]
        volumen_anterior = ultima_posicion.volume
        profit_actual = ultima_posicion.profit
        
        print(f"   📉 Posiciones Activas ({asset}): {num_posiciones} | PnL Latente orden previa: ${profit_actual:.2f} USD")
        
        if profit_actual < 0:
            # Escudo de Drawdown: Cortamos exposición al 50%
            raw_lot_size = volumen_anterior * 0.5
            print(f"   🔪 Estado: DRAWDOWN. Mitigando riesgo estricto (Anti-Martingala al 50%).")
        else:
            # Scaling-In Fraccional: Añadimos 75% del lote anterior para proteger Precio Promedio
            raw_lot_size = volumen_anterior * 0.75 
            print(f"   📈 Estado: PROFIT. Piramidación Fraccional de seguridad (75%).")
            
    else:
        # Primera Operación (Risk-based Allocation)
        capital_en_riesgo_maximo = balance * MAX_GLOBAL_PORTFOLIO_RISK_PCT
        peso_activo = ASSET_VOLATILITY_WEIGHTS.get(asset, 0.15)
        riesgo_base_monetario = capital_en_riesgo_maximo * peso_activo
        riesgo_operacion_usd = riesgo_base_monetario * multiplicador_lote
        
        ticks_at_risk = sl_distance / tick_size
        raw_lot_size = riesgo_operacion_usd / (ticks_at_risk * tick_value) if (ticks_at_risk > 0 and tick_value > 0) else 0.0
            
        print(f"   💰 Balance: ${balance:,.2f} | Margen Libre: ${margen_libre:,.2f}")
        print(f"   ⚖️ Nueva Operación | IA: {tier_text} | Riesgo Aut: ${riesgo_operacion_usd:.2f} USD")

    # =====================================================================
    # 4. NORMALIZACIÓN AL CONTRATO DEL BRÓKER
    # =====================================================================
    if raw_lot_size > 0:
        raw_lot_size = round(raw_lot_size / volume_step) * volume_step
        lot_size = max(min_volume, min(raw_lot_size, max_volume))
        step_decimals = len(str(volume_step).split('.')[1]) if '.' in str(volume_step) else 0
        lot_size = round(lot_size, step_decimals)
    else:
        lot_size = 0.0

    print(f"   📊 Lote Final Normalizado (MT5): {lot_size} | TP Asignado: {tp_mult}x ATR")
    
    parametros_calculados = {"lot_size": lot_size, "stop_loss": float(sl), "take_profit": float(tp)}
    return {**parametros_calculados, "risk_params": parametros_calculados}

# 7. Agente 7: Execution Agent
def execution_agent(state: TradingState):
    signal = state["final_signal"]
    asset_yahoo = state['asset']
    lot_size = state.get("risk_params", {}).get("lot_size", 0.0)
    
    if signal == "HOLD" or lot_size <= 0:
        return {"final_execution": "⏸️ SIN OPERACIÓN: El modelo determinó HOLD o límite de riesgo alcanzado."}    
    
    print(f"🚀 [Execution Agent] Conectando con MT5 para operar {asset_yahoo}...")
    symbol_mt5 = ASSET_MAPPING.get(asset_yahoo, asset_yahoo.replace("-", "").replace("=X", "").lower())
    
    # La validación de inicialización ya la hacemos globalmente, pero revisamos info del símbolo
    symbol_info = mt5.symbol_info(symbol_mt5)
    if symbol_info is None:
        return {"final_execution": f"❌ ERROR: Símbolo '{symbol_mt5}' no encontrado en MT5."}
        
    tipo_relleno_broker = symbol_info.filling_mode
    if tipo_relleno_broker == 1: filling_mode = mt5.ORDER_FILLING_FOK
    elif tipo_relleno_broker == 2: filling_mode = mt5.ORDER_FILLING_IOC
    else: filling_mode = mt5.ORDER_FILLING_FOK

    tick = mt5.symbol_info_tick(symbol_mt5)
    if tick is None:
        return {"final_execution": f"❌ ERROR: No hay cotización actual para '{symbol_mt5}'."}
        
    price = tick.ask if signal == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": float(state['risk_params']['stop_loss']),
        "tp": float(state['risk_params']['take_profit']),
        "deviation": 20,
        "magic": 202601,
        "comment": "TFM Agent",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode, 
    }

    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        resultado = f"❌ ERROR ENVIANDO ORDEN: {result.comment} (Código: {result.retcode})"
    else:
        resultado = (
            f"✅ ÉXITO | ORDEN EJECUTADA EN MT5\n"
            f"   Ticket: {result.order}\n"
            f"   Activo: {symbol_mt5} | Tipo: {signal}\n"
            f"   Precio Ejecución: {result.price}\n"
            f"   Stop Loss: {request['sl']:.5f} | Take Profit: {request['tp']:.5f}"
        )
        
        # =========================================================
        # NUEVO: LLAMADA CORRECTA AL LOGGER DE BACKTEST EN VIVO
        # =========================================================
        log_trade_execution(
            asset=asset_yahoo,
            ml_signal=state.get("ml_prediction", "HOLD"),
            ml_confidence=state.get("ml_confidence", 0.0),
            accion_final=signal,
            ticket=result.order,
            precio_ejecucion=result.price,
            lot_size=lot_size
        )
        
    return {"final_execution": resultado}


# ==============================================================================
# ENSAMBLAJE DEL GRAFO (LANGGRAPH)
# ==============================================================================
workflow = StateGraph(TradingState)
workflow.add_node("market_data", market_data_agent)
workflow.add_node("technical_analyst", technical_analyst_agent)
workflow.add_node("fundamental_analyst", fundamental_analyst_agent)
workflow.add_node("quant_ml", quant_ml_agent)
workflow.add_node("portfolio_manager", portfolio_manager_agent)
workflow.add_node("risk_manager", risk_manager_agent)
workflow.add_node("execution", execution_agent)

workflow.add_edge("market_data", "technical_analyst")
workflow.add_edge("technical_analyst", "fundamental_analyst")
workflow.add_edge("fundamental_analyst", "quant_ml")
workflow.add_edge("quant_ml", "portfolio_manager")
workflow.add_edge("portfolio_manager", "risk_manager")
workflow.add_edge("risk_manager", "execution")
workflow.add_edge("execution", END)

workflow.set_entry_point("market_data")
app = workflow.compile()


# ==============================================================================
# EJECUCIÓN MULTIACTIVO (ORQUESTADOR PRINCIPAL)
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 INICIANDO ECOSISTEMA DE AGENTES MULTIACTIVO")
    print("="*50)
    
    # NUEVO: INICIALIZACIÓN GLOBAL DE MT5. EVITA COLAPSOS EN EL BUCLE.
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")

    if not mt5.initialize(path=path):
        print(f"❌ ERROR CRÍTICO: Fallo al inicializar MT5. Código: {mt5.last_error()}")
        exit()

    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        print(f"❌ ERROR CRÍTICO: Fallo de login en MT5. Código: {mt5.last_error()}")
        mt5.shutdown()
        exit()
    
    # Iteramos sobre cada activo
    for asset_yahoo, asset_mt5 in ASSET_MAPPING.items():
        print(f"\n{'='*40}")
        print(f"🌟 INICIANDO ANÁLISIS PARA: {asset_yahoo} -> {asset_mt5}")
        print(f"{'='*40}")
        
        if not is_market_open(asset_yahoo):
            print(f"   💤 MERCADO CERRADO para {asset_yahoo}. Ahorrando poder computacional.")
            continue 
        
        initial_state = {"asset": asset_yahoo}
        
        try:
            result = app.invoke(initial_state)
            print(f"\n--- RESULTADO FINAL PARA {asset_yahoo} ---")
            print(result["final_execution"])
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO procesando {asset_yahoo}: {e}")
            
    # CIERRE EDUCADO DE CONEXIÓN AL FINALIZAR TODO EL PORTAFOLIO
    mt5.shutdown()
    print("\n✅ ANÁLISIS DEL PORTAFOLIO FINALIZADO.")