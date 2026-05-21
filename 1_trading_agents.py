import yfinance as yf
import pandas as pd
import time
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
    fundamental_sentiment: str # NUEVO: Lo que opina el LLM
    final_signal: str          # NUEVO: La decisión final consensuada
    risk_params: dict   
    final_execution: str

# 2. Agente 1: Extractor de Datos (El Data Engineer)
def market_data_agent(state: TradingState):
    print(f"📡 [Market Data Agent] Obteniendo datos para {state['asset']}...")
    ticker = yf.Ticker(state["asset"])
    hist = ticker.history(period="3mo", interval="1h")
    
    current_price = float(hist['Close'].iloc[-1])
    
    # Actualizamos el estado con los datos crudos
    return {"current_price": current_price, "historical_data": hist}

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
    ema_50 = df.ta.ema(length=50)           # Tendencia principal de mediano plazo
    
    technical_indicators = {
        "RSI_14": float(rsi.iloc[-1]),
        "MACD": float(macd.iloc[-1, 0]),          
        "MACD_Signal": float(macd.iloc[-1, 1]),   
        "ATR_14": float(atr.iloc[-1]),
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
    UMBRAL = 0.53
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

# 6. Agente 5: Gestor de Portafolio (El Consenso)
def portfolio_manager_agent(state: TradingState):
    print("👔 [Portfolio Manager] Debatiendo señales para tomar la decisión final...")
    
    ml_signal = state.get("ml_prediction", "HOLD")
    sentiment = state.get("fundamental_sentiment", "NEUTRAL")
    
    tech = state.get("technical_indicators", {})
    rsi = tech.get("RSI_14", 50.0)
    macd = tech.get("MACD", 0.0)
    macd_signal = tech.get("MACD_Signal", 0.0)
    bb_upper = tech.get("BB_Upper", 0.0)
    bb_lower = tech.get("BB_Lower", 0.0)
    ema = tech.get("EMA_50", 0.0)  
    
    current_price = state.get("current_price")
    
    final_signal = "HOLD"
    
    # Lógica de Análisis técnico:
    if ml_signal == "BUY":
        # Ley 1: RSI (Sobrecompra)
        if rsi >= 75:
            print(f"   🛑 VETO: RSI en {rsi:.1f} (Sobrecompra extrema). Compra abortada.")
            return {"final_signal": "HOLD"}
        # Ley 2: Reversión a la Media (Bollinger)
        if current_price >= bb_upper:
            print(f"   🛑 VETO: Precio perforando Banda de Bollinger Superior. Riesgo de reversión.")
            return {"final_signal": "HOLD"}
        # Ley 3: Tendencia Mayor (EMA)
        if ema > 0 and current_price < ema:
            print(f"   🛑 VETO: Prohibido comprar contra la tendencia institucional (Precio < EMA 50).")
            return {"final_signal": "HOLD"}
        # Ley 4: Momentum (MACD)
        if macd < macd_signal:
            print(f"   🛑 VETO: Momentum bajista detectado en MACD. Compra prematura abortada.")
            return {"final_signal": "HOLD"}

    elif ml_signal == "SELL":
        # Ley 1: RSI (Sobreventa)
        if rsi <= 25:
            print(f"   🛑 VETO: RSI en {rsi:.1f} (Sobreventa extrema). Venta abortada.")
            return {"final_signal": "HOLD"}
        # Ley 2: Reversión a la Media (Bollinger)
        if current_price <= bb_lower:
            print(f"   🛑 VETO: Precio perforando Banda de Bollinger Inferior. Riesgo de rebote.")
            return {"final_signal": "HOLD"}
        # Ley 3: Tendencia Mayor (EMA)
        if ema > 0 and current_price > ema:
            print(f"   🛑 VETO: Prohibido vender contra la tendencia institucional (Precio > EMA 50).")
            return {"final_signal": "HOLD"}
        # Ley 4: Momentum (MACD)
        if macd > macd_signal:
            print(f"   🛑 VETO: Momentum alcista detectado en MACD. Venta prematura abortada.")
            return {"final_signal": "HOLD"}
    
    # Lógica de Consenso Estricto Institucional:
    # Solo operamos si los números (ML) y las noticias (Fundamental) están de acuerdo.
    if ml_signal == "BUY" and sentiment in ["BULLISH", "NEUTRAL"]:
        final_signal = "BUY"
    elif ml_signal == "SELL" and sentiment in ["BEARISH", "NEUTRAL"]:
        final_signal = "SELL"
    else:
        # Si el ML dice BUY pero las noticias son BEARISH, abortamos.
        print(f"   ⚠️ Conflicto detectado (Quant: {ml_signal} | Noticias: {sentiment}). Abortando operación.")
        final_signal = "HOLD"
        
    print(f"   Decisión Final: {final_signal}")
    return {"final_signal": final_signal}

# 7. Agente 6: Gestor de Riesgo (Market Risk)
def risk_manager_agent(state: TradingState):
    print("🛡️ [Risk Manager] Calculando Riesgo Dinámico (ATR y VaR)...")
    price = state["current_price"]
    signal = state["ml_prediction"]
    
    if signal == "HOLD":
         return {"risk_params": {"stop_loss": 0.0, "take_profit": 0.0, "VaR_95": 0.0}}
         
    df = state["historical_data"].copy()
    atr = state["technical_indicators"]["ATR_14"]
    
    # 1. Cálculo del Value at Risk (VaR) Histórico-Paramétrico (95% de confianza)
    df['Returns'] = df['Close'].pct_change()
    volatility = df['Returns'].std()
    var_95_pct = norm.ppf(0.95) * volatility  # ~1.645 desviaciones estándar
    var_price_impact = price * var_95_pct
    
    # 2. SL Dinámico: Elegimos el mayor riesgo entre el ATR y el VaR para protegernos
    atr_multiplier = 1.5 
    rr_ratio = 2.0 
    
    risk_distance = max(atr * atr_multiplier, var_price_impact)
    
    if signal == "BUY":
        sl = price - risk_distance
        tp = price + (risk_distance * rr_ratio)
    elif signal == "SELL":
        sl = price + risk_distance
        tp = price - (risk_distance * rr_ratio)
        
    return {
        "risk_params": {
            "stop_loss": sl, 
            "take_profit": tp,
            "VaR_95_perc": round(var_95_pct * 100, 2),
            "ATR_Value": round(atr, 2)
        }
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
        "volume": 0.01, # Lote mínimo (Ajustar si Forex en Oanda pide 1000, 10000, etc.)
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