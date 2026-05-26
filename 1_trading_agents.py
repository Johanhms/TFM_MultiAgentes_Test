import yfinance as yf
import pandas as pd
import time
import csv
import requests
import xml.etree.ElementTree as ET
import json
import pytz
from datetime import datetime
from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas_ta as ta
import numpy as np
from scipy.stats import norm
import joblib
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- CONFIGURACIÓN DE ACTIVOS (FLEXIBILIDAD MULTIACTIVO) ---
# Clave: Símbolo para descargar datos (Yahoo Finance)
# Valor: Símbolo exacto para ejecutar órdenes (MetaTrader 5)
ASSET_MAPPING = {
    "BTC-USD": "BTCUSD",       # Cripto (sin sufijo en tu Oanda)
    "EURUSD=X": "EURUSD.sml",  # Forex (con sufijo en tu Oanda)
    "GBPUSD=X": "GBPUSD.sml",  # Forex (con sufijo en tu Oanda)
    "GC=F": "XAUUSD.sml",      # Ejemplo futuro: Oro
    "^GSPC": "US500",      # Ejemplo futuro: S&P 500
    "CL=F": "USOIL.sml",       # NUEVO - Materia Prima: Petróleo WTI (Crude Oil)
    "^DJI": "US30",            # NUEVO - Índice: Dow Jones Industrial Average
    "NVDA": "NVDA_CFD.US"      # NUEVO - Acción: Nvidia Corporation
}

CURRENCY_MAP = {
    "BTC-USD": ["USD"],
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "GC=F": ["USD"],
    "^GSPC": ["USD"],
    "CL=F": ["USD"],
    "^DJI": ["USD"],
    "NVDA": ["USD"]
}

# --- MATRIZ DE RIESGO Y DIRECCIÓN DE PORTAFOLIO ---
# Ponderación inversa a la volatilidad del activo (Capacidad de asignación presupuestaria)
ASSET_VOLATILITY_WEIGHTS = {
    "BTC-USD": 0.10,       # Cripto: Mínima asignación por volatilidad extrema
    "NVDA": 0.15,          # Acciones: Baja-Media asignación
    "GC=F": 0.20,          # Materias Primas (Oro): Asignación Media
    "CL=F": 0.20,          # Materias Primas (Petróleo): Asignación Media
    "^GSPC": 0.25,         # Índices (S&P 500): Asignación Media-Alta
    "^DJI": 0.25,          # Índices (Dow Jones): Asignación Media-Alta
    "EURUSD=X": 0.30,      # Forex: Máxima asignación por estabilidad y liquidez
    "GBPUSD=X": 0.30       # Forex: Máxima asignación
}

# --- MATRIZ DE SENSIBILIDAD FUNDAMENTAL ---
ASSET_FUNDAMENTAL_SENSITIVITY = {
    "EURUSD=X": "HIGH",    # Forex: Dependencia total de macro
    "GBPUSD=X": "HIGH",    # Forex: Dependencia total de macro
    "GC=F": "HIGH",        # Oro: Muy sensible a noticias de inflación
    "CL=F": "HIGH",        # Petróleo: Sensible a geopolítica
    "^GSPC": "MEDIUM",     # S&P 500: Sensibilidad media
    "^DJI": "MEDIUM",      # Dow Jones: Sensibilidad media
    "BTC-USD": "LOW",      # Cripto: Domina el análisis técnico/quant
    "NVDA": "LOW"          # Acciones: Domina el modelo predictivo quant
}

# Límites estrictos para evitar sobreexposición total del fondo de inversión
MAX_GLOBAL_PORTFOLIO_RISK_PCT = 0.10  # El riesgo sumado jamás superará el 10% del capital

mt5.initialize()

def is_market_open(asset: str) -> bool:
    """
    Reloj interno que verifica la disponibilidad del mercado basándose 
    en el horario oficial de Nueva York (EST/EDT).
    """
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    
    # 1. Criptomonedas (Nunca duermen)
    if asset == "BTC-USD":
        return True
        
    # Bloqueo General de Fines de Semana para mercados tradicionales
    # Sábado completo, y Domingo antes de las 17:00 NY
    if ny_time.weekday() == 5: 
        return False
    if ny_time.weekday() == 6 and ny_time.hour < 17: 
        return False
        
    # 2. Acciones e Índices Americanos (09:30 a 16:00 NY)
    if asset in ["NVDA", "^DJI", "^GSPC"]:
        if ny_time.hour < 9 or ny_time.hour >= 16:
            return False
        if ny_time.hour == 9 and ny_time.minute < 30:
            return False
        return True
        
    # 3. Forex y Materias Primas (Oro, Petróleo)
    # Operan 24 horas de lunes a viernes, excepto el "Rollover" diario a las 17:00 NY
    if asset in ["EURUSD=X", "GBPUSD=X", "GC=F", "CL=F"]:
        if ny_time.hour == 17: 
            return False
        return True
        
    return True

def get_macro_calendar(currencies: list) -> list:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    eventos = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
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
                evento_txt = f"- [{impact}] {country}: {title} ({date_str} {time_str})"
                eventos.append(evento_txt)
        return eventos[:5]
    except:
        return []

# 1. Definimos el "Estado" que los agentes se irán pasando
# Aquí es donde la estructura es escalable. Luego puedes agregar más campos.
class TradingState(TypedDict):
    asset: str
    current_price: float
    historical_data: pd.DataFrame
    technical_indicators: dict
    ml_prediction: str  
    ml_confidence: float
    fundamental_sentiment: str # NUEVO: Lo que opina el LLM
    final_signal: str          # NUEVO: La decisión final consensuada
    risk_params: dict   
    final_execution: str

# 2. Agente 1: Extractor de Datos (El Data Engineer)
def market_data_agent(state: TradingState):
    print(f"📡 [Market Data Agent] Obteniendo datos en vivo desde MT5 para {state['asset']}...")
    
    asset = state["asset"]
    
    # Usamos el mapeo para saber cómo se llama el activo en tu bróker
    ASSET_MAPPING = {
        "BTC-USD": "BTCUSD", "EURUSD=X": "EURUSD.sml", "GBPUSD=X": "GBPUSD.sml",
        "GC=F": "XAUUSD.sml", "^GSPC": "US500", "CL=F": "USOIL.sml", 
        "^DJI": "US30", "NVDA": "NVDA_CFD.US"
    }
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    # 1. Asegurar conexión a MT5
    if not mt5.initialize():
        print("   ❌ Error: No se pudo conectar a MT5 para descargar velas históricas.")
        return {"historical_data": pd.DataFrame(), "current_price": 0.0}

    # 2. Descargar las últimas 100 velas de 1 Hora directamente del bróker
    # Esto garantiza que el RSI, MACD y EMA se calculen con la gráfica real que tú estás viendo
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 200)
    
    if velas_mt5 is None or len(velas_mt5) == 0:
        print(f"   ⚠️ No se pudieron obtener datos de MT5 para {symbol_mt5}.")
        return {"historical_data": pd.DataFrame(), "current_price": 0.0}
        
    # 3. Transformar los datos de MT5 al formato que Pandas (y tu IA) entienden
    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume'
    }, inplace=True)
    
    # Establecer el tiempo como índice temporal
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 4. Obtener el precio exacto de cierre (o Ask/Bid en vivo)
    current_price = float(df['Close'].iloc[-1])
    
    return {
        "historical_data": df,
        "current_price": current_price
    }

# 3. Agente 2: Analista Técnico Basado en Robustp
def technical_analyst_agent(state: TradingState):
    print("📊 [Technical Analyst] Calculando MACD, RSI y ATR...")
    df = state["historical_data"].copy()
    
    # Cálculos con pandas-ta
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    roc = df.ta.roc(length=10)
    
    bbands = df.ta.bbands(length=20, std=2) # Bandas de Bollinger (2 Desviaciones Estándar)
    ema_50 = df.ta.ema(length=50)            # Tendencia principal de mediano plazo
    df['EMA_50'] = df.ta.ema(length=50)         # Tendencia principal de mediano plazo
    
    # NUEVO: Calculamos la pendiente (momentum) de la EMA comparando con 3 horas atrás
    df['EMA_50_Slope'] = df['EMA_50'].diff(periods=3)
    
    technical_indicators = {
        "RSI_14": float(rsi.iloc[-1]),
        "MACD": float(macd.iloc[-1, 0]),          
        "MACD_Signal": float(macd.iloc[-1, 1]),   
        "ATR_14": float(atr.iloc[-1]),
        "EMA_50": df['EMA_50'].iloc[-1],
        "EMA_50_Slope": df['EMA_50_Slope'].iloc[-1], # Pasamos el dato al gestor
        "BB_Upper": float(bbands.iloc[-1, 2]),    # Banda Superior
        "BB_Lower": float(bbands.iloc[-1, 0]),    # Banda Inferior
        "EMA_50": float(ema_50.iloc[-1]) if pd.notna(ema_50.iloc[-1]) else 0.0             
    }
    
    # Actualizamos el estado con los indicadores
    return {"technical_indicators": technical_indicators}

# 4. Agente 3: El Modelo Cuantitativo (¡Tu futuro Máster TFM!)
def quant_ml_agent(state: TradingState):
    print("🧠 [Quant ML Agent] Consultando modelo de Inteligencia Artificial...")
    
    asset = state["asset"]
    safe_name = state["asset"].replace("=X", "").replace("=F", "").replace("^", "")
    
    BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    model_path = BASE_DIR / f'quant_model_1h_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'
    
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el modelo para {safe_name}. Ejecuta train_model.py primero.")
        return {"ml_prediction": "HOLD"}
    
    # 2. Reconstruimos los indicadores del estado actual para igualar el formato del modelo
    df = state["historical_data"].copy()
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    roc = df.ta.roc(length=10)
    
    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1
    
    df_live = pd.concat([df, macd, rsi, atr, roc], axis=1)
    
    # 3. Extraemos EXACTAMENTE la última vela de datos (el momento presente)
    latest_data = df_live[features].iloc[-1:]
    
    # Si hay algún NaN (porque faltan datos para calcular un indicador largo), abortamos
    if latest_data.isnull().values.any():
        return {"ml_prediction": "HOLD"}
        
    # 4. Predicción Probabilística
    # No pedimos solo 1 o 0, pedimos la probabilidad matemática de que suba o baje
    probabilities = model.predict_proba(latest_data)[0]
    prob_bajada = probabilities[0] # Probabilidad de que la clase sea 0
    prob_subida = probabilities[1] # Probabilidad de que la clase sea 1
    
    # 5. Lógica de Decisión con Umbrales (Thresholds)
    # Exigimos un 60% de seguridad matemática para arriesgar capital.
    # Si el modelo está indeciso (ej. 52% vs 48%), forzamos un HOLD.
    UMBRAL = 0.52
    if prob_subida >= UMBRAL:
        prediction = "BUY"
        confianza = prob_subida
    elif prob_bajada > UMBRAL:
        prediction = "SELL"
        confianza = prob_bajada
    else:
        prediction = "HOLD"
        confianza = max(prob_subida, prob_bajada)
        
    print(f"   Predicción: {prediction} (Confianza del modelo: {confianza*100:.1f}%)")
        
    return {"ml_prediction": prediction, "ml_confidence": confianza}

# 5. Agente 4: Analista Fundamental (NLP & LLMs)
def fundamental_analyst_agent(state: TradingState):
    asset = state["asset"]
    print(f"📰 [Fundamental Agent] Evaluando {asset}...")
    
    # 1. SISTEMA DE CACHÉ: Revisar si ya hicimos este trabajo hoy
    hoy = datetime.now().strftime("%Y-%m-%d")
    archivo_cache = "memoria_fundamental.json"
    memoria = {}
    
    if os.path.exists(archivo_cache):
        try:
            with open(archivo_cache, "r") as f:
                memoria = json.load(f)
        except Exception:
            pass
            
    # Si la memoria es de hoy y ya tenemos el sentimiento de este activo, lo usamos instantáneamente
    if memoria.get("fecha") == hoy and asset in memoria.get("sentimientos", {}):
        sentimiento_guardado = memoria["sentimientos"][asset]
        print(f"   ⚡ Usando memoria rápida del día: {sentimiento_guardado} (Evitando API)")
        return {"fundamental_sentiment": sentimiento_guardado}

    # =====================================================================
    # 2. SI NO ESTÁ EN CACHÉ: Hacemos el trabajo pesado (1 vez al día)
    # =====================================================================
    print(f"   📥 Descargando Macro y Noticias frescas para hoy...")
    
    # Extraer Noticias
    ticker = yf.Ticker(asset)
    news_list = ticker.news
    headlines = []
    if news_list:
        for n in news_list[:3]:
            if isinstance(n, dict):
                title = n.get('title') or (n.get('content', {}).get('title') if 'content' in n else None)
                if title: headlines.append(f"- {title}")
                
    news_text = "\n".join(headlines) if headlines else "Sin noticias."
    
    # Extraer Calendario Macro
    divisas = CURRENCY_MAP.get(asset, ["USD"])
    eventos_macro = get_macro_calendar(divisas)
    macro_text = "\n".join(eventos_macro) if eventos_macro else "Sin eventos macro."
    
    # Consultar a Gemini
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        prompt = f"""
        Activo: {asset}.
        Noticias: {news_text}
        Macro: {macro_text}
        Sintetiza y responde ÚNICAMENTE con una palabra: BULLISH, BEARISH o NEUTRAL.
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
    
    # 3. ACTUALIZAR CACHÉ PARA EL RESTO DEL DÍA
    if memoria.get("fecha") != hoy:
        memoria = {"fecha": hoy, "sentimientos": {}}
        
    memoria["sentimientos"][asset] = sentiment
    
    with open(archivo_cache, "w") as f:
        json.dump(memoria, f, indent=4)
        
    # Pausa de seguridad solo si usamos la API para no saturar al pasar al siguiente activo
    time.sleep(3) 
    
    return {"fundamental_sentiment": sentiment}

def registrar_veto_csv(asset: str, ml_signal: str, motivo: str, precio: float, rsi: float, ema: float):
    """Guarda un registro tabular de cada vez que el Gestor de Portafolio bloquea a la IA."""
    archivo = "tfm_auditoria_vetos.csv"
    file_exists = os.path.isfile(archivo)
    
    with open(archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Creamos las columnas estructurales para tu posterior análisis
            writer.writerow(["Fecha", "Activo", "Señal_ML_Bloqueada", "Motivo_Veto", "Precio_Evitado", "Nivel_RSI", "Nivel_EMA50"])
        
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([fecha_actual, asset, ml_signal, motivo, round(precio, 5), round(rsi, 2), round(ema, 5)])

# 6. Agente 5: Gestor de Portafolio (El Consenso)
def portfolio_manager_agent(state: TradingState):
    print("👔 [Portfolio Manager] Debatiendo señales para tomar la decisión final...")
    
    asset = state.get("asset", "UNKNOWN")
    ml_signal = state.get("ml_prediction", "HOLD")
    ml_confidence = state.get("ml_confidence", 0.0)
    sentiment = state.get("fundamental_sentiment", "NEUTRAL")
    current_price = state.get("current_price")
    
    tech = state.get("technical_indicators", {})
    rsi = tech.get("RSI_14", 50.0)
    macd = tech.get("MACD", 0.0)
    macd_signal = tech.get("MACD_Signal", 0.0)
    bb_upper = tech.get("BB_Upper", 0.0)
    bb_lower = tech.get("BB_Lower", 0.0)
    ema = tech.get("EMA_50", 0.0)    
    ema_slope = tech.get("EMA_50_Slope", 0.0)
    
    final_signal = "HOLD"
    
    # 1. LEYES DE VETO TÉCNICO INQUEBRANTABLES (Ahora con Prints Visibles)
    if ml_signal == "BUY":
        if rsi >= 75 or current_price >= bb_upper or macd < macd_signal:
            motivo = "Veto Técnico Alcista (RSI/BB/MACD)"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, ema)
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}
            
        if ema > 0 and current_price < ema and ema_slope < 0:
            motivo = "Precio bajo EMA y Pendiente Bajista"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, ema)
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}

    elif ml_signal == "SELL":
        if rsi <= 25 or current_price <= bb_lower or macd > macd_signal:
            motivo = "Veto Técnico Bajista (RSI/BB/MACD)"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, ema)
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}
            
        if ema > 0 and current_price > ema and ema_slope > 0:
            motivo = "Precio sobre EMA y Pendiente Alcista"
            print(f"   🛑 VETO: {motivo}")
            registrar_veto_csv(asset, ml_signal, motivo, current_price, rsi, ema)
            return {"final_signal": "HOLD", "ml_confidence": ml_confidence}

    # 2. LÓGICA DE CONSENSO FUNDAMENTAL POR SENSIBILIDAD
    sensitivity = ASSET_FUNDAMENTAL_SENSITIVITY.get(asset, "HIGH")
    final_signal = ml_signal

    if sensitivity == "HIGH":
        # Forex y Materias Primas: Exige consenso estricto
        if ml_signal == "BUY" and sentiment == "BEARISH":
            final_signal = "HOLD"
            print(f"   ⚠️ Conflicto Macro (Sensibilidad ALTA): Quant dice BUY, Noticias dicen BEARISH. Abortando.")
        elif ml_signal == "SELL" and sentiment == "BULLISH":
            final_signal = "HOLD"
            print(f"   ⚠️ Conflicto Macro (Sensibilidad ALTA): Quant dice SELL, Noticias dicen BULLISH. Abortando.")
            
    elif sensitivity == "MEDIUM":
        # Índices: Tolera neutralidad, pero veta conflictos directos severos
        if (ml_signal == "BUY" and sentiment == "BEARISH") or (ml_signal == "SELL" and sentiment == "BULLISH"):
            final_signal = "HOLD"
            print(f"   ⚠️ Conflicto Macro (Sensibilidad MEDIA): Divergencia severa detectada. Abortando.")
            
    elif sensitivity == "LOW":
        # Acciones y Criptos: El modelo Quant domina. Las noticias se ignoran direccionalmente.
        if ml_signal != "HOLD":
            print(f"   🛡️ Sensibilidad Fundamental BAJA: Priorizando predicción Quant ({ml_signal}).")
            final_signal = ml_signal

    if final_signal != "HOLD":
        print(f"   ✅ Ecosistema Alineado. Permiso de ejecución concedido.")
        
    print(f"   Decisión Final: {final_signal} | Confianza Asociada: {ml_confidence*100:.1f}%")
    return {"final_signal": final_signal, "ml_confidence": ml_confidence}

# 7. Agente 6: Gestor de Riesgo (Market Risk)
def risk_manager_agent(state: TradingState):
    print("🛡️ [Risk Manager] Calculando dimensionamiento de posición y límites de cuenta...")
    
    final_signal = state.get("final_signal", "HOLD")
    asset = state.get("asset")
    confianza = state.get("ml_confidence", 0.0)
    tech = state.get("technical_indicators", {})
    atr = tech.get("ATR_14", 0.0)
    
    if final_signal == "HOLD" or atr <= 0:
        vacio = {"lot_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        return {**vacio, "risk_params": vacio}
    
    # Mapeo local para identificar el símbolo exacto en tu MetaTrader 5
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
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
        
    # 1. Conexión en tiempo real al balance del bróker (Oanda vía MT5)
    if not mt5.initialize():
        print("   ❌ Error crítico: MT5 no responde. Abortando cálculo de riesgo.")
        vacio = {"lot_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        return {**vacio, "risk_params": vacio}
    
    account_info = mt5.account_info()
    balance = account_info.balance if account_info is not None else 10000.0
        
    # =====================================================================
    # 2. PARÁMETROS OPERATIVOS BASE (Anclados al precio en vivo de MT5)
    # =====================================================================
    distance = atr * 1.5
    
    # Extraemos el tick en vivo (milisegundo actual) directamente del bróker
    tick = mt5.symbol_info_tick(symbol_mt5)
    
    if tick is None:
        print(f"   ⚠️ No se pudo obtener el precio en vivo de MT5 para {symbol_mt5}. Abortando.")
        vacio = {"lot_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        return {**vacio, "risk_params": vacio}

    # Calculamos SL y TP desde el precio real al que se ejecutará la orden
    if final_signal == "BUY":
        precio_ejecucion_real = tick.ask
        sl = precio_ejecucion_real - distance
        tp = precio_ejecucion_real + (distance * 2.0)
    else:
        precio_ejecucion_real = tick.bid
        sl = precio_ejecucion_real + distance
        tp = precio_ejecucion_real - (distance * 2.0)
        
    # 3. Extraer especificaciones del bróker para normalizar
    symbol_info = mt5.symbol_info(symbol_mt5)
    if symbol_info is None:
        print(f"   ⚠️ Símbolo {symbol_mt5} no detectado. Abortando.")
        vacio = {"lot_size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        return {**vacio, "risk_params": vacio}
        
    volume_step = symbol_info.volume_step
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value 

    # =====================================================================
    # 🧠 LÓGICA DE BIFURCACIÓN DE RIESGO (La regla solicitada)
    # =====================================================================
    posiciones_activo = mt5.positions_get(symbol=symbol_mt5)
    
    if posiciones_activo and len(posiciones_activo) > 0:
        # CAMINO A: Ya hay operaciones abiertas. Aplicamos reducción física estricta a la mitad.
        # Ordenamos por tiempo para asegurar que tomamos la más reciente
        ultima_posicion = sorted(posiciones_activo, key=lambda p: p.time)[-1]
        volumen_anterior = ultima_posicion.volume
        
        # Reducción estricta solicitada: 50% del lote anterior
        raw_lot_size = volumen_anterior * 0.5
        
        print(f"   📉 Posiciones Activas ({asset}): {len(posiciones_activo)}")
        print(f"   🔪 Lote Anterior: {volumen_anterior} | Mitigando al 50% por Regla de Riesgo Secuencial.")
        
    else:
        # CAMINO B: Es la primera operación. Calculamos desde cero en base al Capital y la IA.
        capital_en_riesgo_maximo = balance * MAX_GLOBAL_PORTFOLIO_RISK_PCT
        peso_activo = ASSET_VOLATILITY_WEIGHTS.get(asset, 0.15)
        riesgo_base_monetario = capital_en_riesgo_maximo * peso_activo
        
        if 0.50 <= confianza <= 0.529:
            multiplicador_lote = 0.5
            tier_text = "LOTE MÍNIMO"
        elif 0.53 <= confianza <= 0.539:
            multiplicador_lote = 1.0
            tier_text = "LOTE MEDIO"
        elif confianza >= 0.54:
            multiplicador_lote = 1.5
            tier_text = "LOTE MAYOR"
        else:
            multiplicador_lote = 0.5
            tier_text = "LOTE DE MITIGACIÓN"

        riesgo_operacion_usd = riesgo_base_monetario * multiplicador_lote
        ticks_at_risk = distance / tick_size
        
        if ticks_at_risk > 0 and tick_value > 0:
            raw_lot_size = riesgo_operacion_usd / (ticks_at_risk * tick_value)
        else:
            raw_lot_size = 0.0
            
        print(f"   💰 Balance de Cuenta: ${balance:,.2f} USD")
        print(f"   ⚖️ Primera Operación | Clasificación IA: {tier_text} | Peso Asignado: {peso_activo*100}%")
        print(f"   🛡️ Riesgo Monetario Autorizado: ${riesgo_operacion_usd:.2f} USD")

    # =====================================================================
    # 4. NORMALIZACIÓN ESTRICTA DE MT5
    # =====================================================================
    # Ajustamos al salto del bróker (ej. de 0.18 a 0.18 o 0.19)
    if raw_lot_size > 0:
        raw_lot_size = round(raw_lot_size / volume_step) * volume_step
        lot_size = max(min_volume, min(raw_lot_size, max_volume))
        step_decimals = len(str(volume_step).split('.')[1]) if '.' in str(volume_step) else 0
        lot_size = round(lot_size, step_decimals)
    else:
        lot_size = 0.0

    print(f"   📊 Tamaño de Lote Corregido para MT5: {lot_size} lotes")
    
    parametros_calculados = {
        "lot_size": lot_size,
        "stop_loss": float(sl),
        "take_profit": float(tp)
    }
    
    return {
        **parametros_calculados,
        "risk_params": parametros_calculados
    }

# 8. Agente 7: Ejecutor
def execution_agent(state: TradingState):
    signal =  state["final_signal"]
    
    if signal == "HOLD":
        return {"final_execution": "⏸️ SIN OPERACIÓN: El modelo determinó HOLD."}    
    
    print(f"🚀 [Execution Agent] Conectando con MT5 para operar {state['asset']}...")
    
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")

    if not mt5.initialize(path=path):
        return {"final_execution": f"❌ ERROR: Fallo al inicializar MT5. Código: {mt5.last_error()}"}

    authorized = mt5.login(login=login, password=password, server=server)
    if not authorized:
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: Fallo de login en MT5. Código: {mt5.last_error()}"}

    # --- TRADUCCIÓN FLEXIBLE DE SÍMBOLOS ---
    asset_yahoo = state['asset']
    if asset_yahoo in ASSET_MAPPING:
        symbol_mt5 = ASSET_MAPPING[asset_yahoo]
    else:
        symbol_mt5 = asset_yahoo.replace("-", "").replace("=X", "").lower()
    
    print(f"🔍 Buscando símbolo en MT5: {symbol_mt5}")

    if not mt5.symbol_select(symbol_mt5, True):
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: Símbolo '{symbol_mt5}' no encontrado."}

    # Extraemos las propiedades estructurales del activo en el bróker
    symbol_info = mt5.symbol_info(symbol_mt5)
    if symbol_info is None:
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: No hay información para '{symbol_mt5}'."}
        
    # =====================================================================
    # NUEVO: DETECCIÓN DINÁMICA DE FILLING MODE (Soporte Multiactivo)
    # =====================================================================
    tipo_relleno_broker = symbol_info.filling_mode
    
    if tipo_relleno_broker == 1:
        filling_mode = mt5.ORDER_FILLING_FOK
    elif tipo_relleno_broker == 2:
        filling_mode = mt5.ORDER_FILLING_IOC
    else:
        # Si el bróker devuelve 3 (soporta ambos), priorizamos FOK por seguridad en Oanda
        filling_mode = mt5.ORDER_FILLING_FOK

    tick = mt5.symbol_info_tick(symbol_mt5)
    if tick is None:
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: No hay cotización actual para '{symbol_mt5}'."}
        
    signal = state["ml_prediction"]
    price = tick.ask if signal == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    # Construir petición con el modo correcto
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": float(state["risk_params"]["lot_size"]),
        "type": order_type,
        "price": price,
        "sl": float(state['risk_params']['stop_loss']),
        "tp": float(state['risk_params']['take_profit']),
        "deviation": 20,
        "magic": 202601,
        "comment": "TFM Agent",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode, # <--- La variable inteligente
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
    
    mt5.shutdown()
    
    return {"final_execution": resultado}

# --- ENSAMBLAJE DEL GRAFO (LANGGRAPH) ---

# Iniciamos el grafo con nuestro Estado
workflow = StateGraph(TradingState)

# Añadimos los nodos (nuestros agentes)
workflow.add_node("market_data", market_data_agent)
workflow.add_node("technical_analyst", technical_analyst_agent)
workflow.add_node("fundamental_analyst", fundamental_analyst_agent)
workflow.add_node("quant_ml", quant_ml_agent)
workflow.add_node("portfolio_manager", portfolio_manager_agent)
workflow.add_node("risk_manager", risk_manager_agent)
workflow.add_node("execution", execution_agent)

# Definimos el flujo (la tubería)
workflow.add_edge("market_data", "technical_analyst")
workflow.add_edge("technical_analyst", "fundamental_analyst")
workflow.add_edge("fundamental_analyst", "quant_ml")
workflow.add_edge("quant_ml", "portfolio_manager")
workflow.add_edge("portfolio_manager", "risk_manager")
workflow.add_edge("risk_manager", "execution")
workflow.add_edge("execution", END)

# Configuramos el punto de entrada
workflow.set_entry_point("market_data")

# Compilamos la aplicación
app = workflow.compile()


# --- EJECUCIÓN MULTIACTIVO ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 INICIANDO ECOSISTEMA DE AGENTES MULTIACTIVO")
    print("="*50)
    
    # Iteramos sobre cada activo definido en la configuración
    for asset_yahoo, asset_mt5 in ASSET_MAPPING.items():
        print(f"\n{'='*40}")
        print(f"🌟 INICIANDO ANÁLISIS PARA: {asset_yahoo} -> {asset_mt5}")
        print(f"{'='*40}")
        
        if not is_market_open(asset_yahoo):
            print(f"   💤 MERCADO CERRADO para {asset_yahoo}. Ahorrando poder computacional.")
            continue  # Salta al siguiente activo inmediatamente
        
        # Estado inicial para el activo en turno
        initial_state = {"asset": asset_yahoo}
        
        try:
            # Ejecutamos el flujo completo para este activo
            result = app.invoke(initial_state)
            
            print(f"\n--- RESULTADO FINAL PARA {asset_yahoo} ---")
            print(result["final_execution"])
            
        except Exception as e:
            # Si un activo falla (ej. error de datos), el ciclo continúa con el siguiente
            print(f"\n❌ ERROR CRÍTICO procesando {asset_yahoo}: {e}")
            
    print("\n✅ ANÁLISIS DEL PORTAFOLIO FINALIZADO.")