import yfinance as yf
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas_ta as ta
import numpy as np
from scipy.stats import norm
import joblib
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
    "GC=F": "XAUUSD.sml",      # Ejemplo futuro: Oro
    "^GSPC": "spx500.sml"      # Ejemplo futuro: S&P 500
}

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
    
    technical_indicators = {
        "RSI_14": float(rsi.iloc[-1]),
        "MACD": float(macd.iloc[-1, 0]),          
        "MACD_Signal": float(macd.iloc[-1, 1]),   
        "ATR_14": float(atr.iloc[-1])             
    }
    
    # Actualizamos el estado con los indicadores
    return {"technical_indicators": technical_indicators}

# 4. Agente 3: El Modelo Cuantitativo (¡Tu futuro Máster TFM!)
def quant_ml_agent(state: TradingState):
    print("🧠 [Quant ML Agent] Consultando modelo de Inteligencia Artificial...")
    
    # 1. Cargamos el modelo pre-entrenado y las features que espera recibir
    try:
        model = joblib.load('quant_model_1h.joblib')
        features = joblib.load('model_features_1h.joblib')
    except FileNotFoundError:
        print("❌ ERROR: No se encontró el modelo. Ejecuta train_model.py primero.")
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
    if prob_subida > 0.50:
        prediction = "BUY"
        confianza = prob_subida
    elif prob_bajada > 0.50:
        prediction = "SELL"
        confianza = prob_bajada
    else:
        prediction = "HOLD"
        confianza = max(prob_subida, prob_bajada)
        
    print(f"   Predicción: {prediction} (Confianza del modelo: {confianza*100:.1f}%)")
        
    return {"ml_prediction": prediction}

# 5. Agente 4: Analista Fundamental (NLP & LLMs)
def fundamental_analyst_agent(state: TradingState):
    print("📰 [Fundamental Agent] Leyendo noticias financieras en tiempo real...")
    ticker = yf.Ticker(state["asset"])
    news_list = ticker.news
    
    headlines = []
    if news_list:
        for n in news_list[:5]:
            if isinstance(n, dict):
                # Búsqueda exhaustiva del título de la noticia
                title = n.get('title')
                if not title and 'content' in n and isinstance(n['content'], dict):
                    title = n['content'].get('title')
                
                if title:
                    headlines.append(f"- {title}")
                    
    if not headlines:
        print("   ⚠️ No se encontraron noticias recientes. Asumiendo NEUTRAL.")
        return {"fundamental_sentiment": "NEUTRAL"}
        
    news_text = "\n".join(headlines)
    
    try:
        # AQUÍ ESTÁ EL CEREBRO DEFINITIVO QUE VALIDAMOS
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        
        prompt = f"""
        Eres un analista financiero experto. Revisa los siguientes titulares recientes sobre {state["asset"]}:
        {news_text}
        
        Determina el sentimiento general del mercado para este activo.
        Responde ÚNICAMENTE con una de estas tres palabras: BULLISH, BEARISH o NEUTRAL.
        No incluyas explicaciones.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        sentiment = response.content.strip().upper()
        
        # Filtro de seguridad por si el LLM añade algún punto final invisible
        if "BULLISH" in sentiment: 
            sentiment = "BULLISH"
        elif "BEARISH" in sentiment: 
            sentiment = "BEARISH"
        else: 
            sentiment = "NEUTRAL"
            
        print(f"   Sentimiento detectado: {sentiment}")
        return {"fundamental_sentiment": sentiment}
        
    except Exception as e:
        print(f"❌ Error en el LLM: {e}")
        print("   Sentimiento por defecto: NEUTRAL")
        return {"fundamental_sentiment": "NEUTRAL"}

# 6. Agente 5: Gestor de Portafolio (El Consenso)
def portfolio_manager_agent(state: TradingState):
    print("👔 [Portfolio Manager] Debatiendo señales para tomar la decisión final...")
    
    ml_signal = state.get("ml_prediction", "HOLD")
    sentiment = state.get("fundamental_sentiment", "NEUTRAL")
    
    final_signal = "HOLD"
    
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
        return {"final_execution": "⏸️ SIN OPERACIÓN: El modelo determinó HOLD (Condiciones de riesgo no favorables)."}    
    
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
    
    # Busca el símbolo en nuestro diccionario de mapeo
    if asset_yahoo in ASSET_MAPPING:
        symbol_mt5 = ASSET_MAPPING[asset_yahoo]
    else:
        # Fallback de seguridad si olvidas agregarlo al diccionario
        symbol_mt5 = asset_yahoo.replace("-", "").replace("=X", "").lower()
    
    print(f"🔍 Buscando símbolo en MT5: {symbol_mt5}")

    # Asegurarnos de que el símbolo esté visible
    if not mt5.symbol_select(symbol_mt5, True):
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: Símbolo '{symbol_mt5}' no encontrado. Revisa si está en el Market Watch de MT5."}

    # Obtener precios del símbolo
    tick = mt5.symbol_info_tick(symbol_mt5)
    if tick is None:
        mt5.shutdown()
        return {"final_execution": f"❌ ERROR: No hay cotización para '{symbol_mt5}'."}
        
    signal = state["ml_prediction"]
    price = tick.ask if signal == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    
    # Construir petición
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": 0.01, # Lote mínimo para BTC en MT5 suele ser 0.01, ajusta si Oanda te pide más
        "type": order_type,
        "price": price,
        "sl": float(state['risk_params']['stop_loss']),
        "tp": float(state['risk_params']['take_profit']),
        "deviation": 20,
        "magic": 202601,
        "comment": "TFM Agent",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, 
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

# --- EJECUCIÓN ---
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