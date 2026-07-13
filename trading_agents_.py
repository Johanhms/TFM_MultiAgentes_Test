import sys
import io
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
import warnings
from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

load_dotenv()

# --- CONFIGURACIÓN DE ACTIVOS Y ESTRATEGIA ---
STRATEGY_TYPE = "LONG_SHORT"  # "LONG_SHORT" o "LONG_ONLY"

ASSET_MAPPING = {
    "BTC-USD": "BTCUSD", "EURUSD=X": "EURUSD.sml", "GBPUSD=X": "GBPUSD.sml",
    "GC=F": "XAUUSD.sml", "^GSPC": "US500", "CL=F": "USOIL.sml", 
    "^DJI": "US30", "NVDA": "NVDA_CFD.US"
}
CURRENCY_MAP = {
    "BTC-USD": ["USD"], "EURUSD=X": ["EUR", "USD"], "GBPUSD=X": ["GBP", "USD"],
    "GC=F": ["USD"], "^GSPC": ["USD"], "CL=F": ["USD"], "^DJI": ["USD"], "NVDA": ["USD"]
}

# --- PARÁMETROS INSTITUCIONALES DE RIESGO ---
MAX_GLOBAL_PORTFOLIO_RISK_PCT = 0.03  
MAX_TRADES_PER_DAY_PER_ASSET = 3      
MAX_DAILY_DRAWDOWN_PCT = 0.02         
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

# --- MONITOREO DE DRIFT DE PROBABILIDADES (conecta con 'prob_baseline' guardado por train_model_.py) ---
PSI_WARNING_THRESHOLD = 0.10   # 0.10-0.25: drift moderado, se avisa pero se sigue operando
PSI_VETO_THRESHOLD = 0.25      # >0.25: drift significativo, se vetea la señal por precaución
PROB_HISTORY_MIN_SAMPLES = 30  # muestras mínimas en vivo antes de confiar en el PSI calculado
PROB_HISTORY_MAX_SAMPLES = 150 # ventana rolling de probabilidades recientes conservadas en disco

# --- RIESGO POR CORRELACIÓN ENTRE ACTIVOS (conecta con 'correlation_matrix_*.joblib' de train_model_.py) ---
MAX_PORTFOLIO_CORRELATED_RISK_PCT = 0.06  # tope de riesgo agregado correlacionado (~2x el riesgo por operación individual)
# Reverse mapping para poder identificar el 'asset' lógico a partir del símbolo MT5 de una posición abierta
REVERSE_ASSET_MAPPING = {v: k for k, v in ASSET_MAPPING.items()}

_correlation_cache = {}
def get_correlation_matrix(suffix: str):
    """Carga (con caché en memoria) la matriz de correlación generada por train_model_OP3.py."""
    if suffix in _correlation_cache:
        return _correlation_cache[suffix]
    path = BASE_DIR / f'correlation_matrix_{suffix}.joblib'
    if not path.exists():
        _correlation_cache[suffix] = None
        return None
    data = joblib.load(path)
    _correlation_cache[suffix] = data['correlation_matrix']
    return _correlation_cache[suffix]

def get_position_risk_usd(pos) -> float:
    """
    Riesgo real en USD de una posición abierta = distancia real a su stop-loss × volumen × valor del tick.
    Si la posición no tiene SL definido, se devuelve 0.0 y se excluye del agregado (no se puede
    cuantificar su riesgo real; considera exigir SL obligatorio en execution_agent si esto te preocupa).
    """
    if not pos.sl or pos.sl == 0:
        return 0.0
    symbol_info = mt5.symbol_info(pos.symbol)
    if not symbol_info or not symbol_info.trade_tick_size:
        return 0.0
    sl_distance = abs(pos.price_open - pos.sl)
    ticks = sl_distance / symbol_info.trade_tick_size
    return ticks * symbol_info.trade_tick_value * pos.volume

def solve_max_correlated_risk(asset: str, candidate_direction: int, existing_risks: dict,
                               corr_matrix, max_portfolio_risk_usd: float):
    """
    Resuelve la varianza CUADRÁTICA de portafolio: Var(w) = w' · Corr · w, donde w es el vector
    de riesgo-en-USD firmado (signo = dirección: +BUY / -SELL) de todas las posiciones, incluida
    la candidata. Encuentra el máximo riesgo (x, en USD) asignable a la operación candidata tal
    que sqrt(Var(w)) no supere max_portfolio_risk_usd.

    existing_risks: {otro_activo: riesgo_usd_firmado} de las posiciones YA abiertas.
    Devuelve (x_max, varianza_existente_A, es_factible).
    """
    assets_existentes = [a for a in existing_risks if a in corr_matrix.columns]

    # A: varianza ya comprometida entre posiciones existentes (sin la candidata)
    A = 0.0
    for a_i in assets_existentes:
        for a_j in assets_existentes:
            A += existing_risks[a_i] * existing_risks[a_j] * corr_matrix.loc[a_i, a_j]

    # B: covarianza cruzada entre la candidata y cada posición existente
    B = 0.0
    if asset in corr_matrix.columns:
        for a_i in assets_existentes:
            B += existing_risks[a_i] * corr_matrix.loc[asset, a_i]

    k = candidate_direction * B
    cap_sq = max_portfolio_risk_usd ** 2
    discriminante = k ** 2 - (A - cap_sq)

    if discriminante < 0:
        return 0.0, A, False  # las posiciones existentes YA superan el cap, ni con riesgo 0 se cumple

    x_max = -k + np.sqrt(discriminante)
    return max(x_max, 0.0), A, True

def compute_psi(baseline_props: np.ndarray, current_props: np.ndarray, epsilon: float = 1e-4) -> float:
    """
    Population Stability Index. Debe coincidir exactamente con la implementación de
    train_model_OP2.py (se duplica aquí a propósito para que este script de producción
    no dependa de que el script de entrenamiento esté presente en el mismo servidor).
    """
    baseline = np.clip(baseline_props, epsilon, None)
    current = np.clip(current_props, epsilon, None)
    return float(np.sum((current - baseline) * np.log(current / baseline)))

def check_probability_drift(safe_name: str, timeframe_suffix: str, prob_buy: float,
                             prob_sell: float, prob_baseline: dict):
    """
    Actualiza el historial rolling de probabilidades en vivo (persistido en disco entre
    ejecuciones) y, cuando hay suficientes muestras, calcula el PSI contra el baseline
    guardado en el bundle del modelo. Devuelve None si el bundle es de una versión antigua
    sin baseline, o si aún no hay historial suficiente.
    """
    if not prob_baseline:
        return None

    hist_path = BASE_DIR / f'prob_history_{safe_name}_{timeframe_suffix}.joblib'
    history = joblib.load(hist_path) if hist_path.exists() else {'prob_buy': [], 'prob_sell': []}
    history['prob_buy'].append(prob_buy)
    history['prob_sell'].append(prob_sell)
    history['prob_buy'] = history['prob_buy'][-PROB_HISTORY_MAX_SAMPLES:]
    history['prob_sell'] = history['prob_sell'][-PROB_HISTORY_MAX_SAMPLES:]
    joblib.dump(history, hist_path)

    if len(history['prob_buy']) < PROB_HISTORY_MIN_SAMPLES:
        return None

    bin_edges = prob_baseline['bin_edges']

    def _hist(values):
        counts, _ = np.histogram(values, bins=bin_edges)
        total = counts.sum()
        return (counts / total) if total > 0 else np.ones(len(counts)) / len(counts)

    psi_buy = compute_psi(prob_baseline['prob_buy_baseline'], _hist(history['prob_buy']))
    psi_sell = compute_psi(prob_baseline['prob_sell_baseline'], _hist(history['prob_sell']))
    return max(psi_buy, psi_sell)

# ==============================================================================
# FUNCIONES DE LOGGING Y SOPORTE
# ==============================================================================
def log_trade_execution(asset: str, ml_signal: str, ml_confidence: float, accion_final: str, ticket: int, precio_ejecucion: float, lot_size: float):
    filename = BASE_DIR / "trading_history_log.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp', 'Asset', 'ML_Prediction', 'ML_Confidence', 'Executed_Action', 'MT5_Ticket', 'Execution_Price', 'Lot_Size'])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), asset, ml_signal, round(ml_confidence, 4), accion_final, ticket, precio_ejecucion, lot_size])

def is_market_open(asset: str) -> bool:
    ny_tz = pytz.timezone('America/New_York')
    ny_time = datetime.now(ny_tz)
    if asset == "BTC-USD": return True
    if ny_time.weekday() == 5: return False
    if ny_time.weekday() == 6 and ny_time.hour < 17: return False
    return True 

# ==============================================================================
# GRAFO DE ESTADO
# ==============================================================================
class TradingState(TypedDict):
    asset: str
    timeframe_suffix: str
    current_price: float
    historical_data: pd.DataFrame
    technical_indicators: dict
    ml_prediction: str  
    ml_confidence: float
    fundamental_sentiment: str 
    final_signal: str          
    risk_params: dict   
    final_execution: str
    drift_alert: bool

# 1. Agente 1: Market Data Agent
def market_data_agent(state: TradingState):
    asset = state["asset"]
    tf_suffix = state.get("timeframe_suffix", "1h")
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    mt5_tf = mt5.TIMEFRAME_D1 if tf_suffix == "1d" else mt5.TIMEFRAME_H1
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5_tf, 0, 300)
    
    # CORRECCIÓN DE NUMPY: Se debe usar verificación de 'None' o evaluar la longitud
    if velas_mt5 is None or len(velas_mt5) == 0: 
        print(f"   ❌ ERROR: Sin datos en MT5 para {symbol_mt5}")
        return state
        
    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    tick = mt5.symbol_info_tick(symbol_mt5)
    current_price = tick.ask if tick else df['Close'].iloc[-1]
    
    print(f"   📊 [Market Data] Cotización actual extraída: {current_price}")
    return {"historical_data": df, "current_price": current_price}

# 2. Agente 2: Technical Analyst
def technical_analyst_agent(state: TradingState):
    if "historical_data" not in state: return state
    df = state["historical_data"].copy()
    asset = state["asset"]
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    # Features Base
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(window=20).std()
    df['Dist_EMA20'] = df['Close'] / df.ta.ema(length=20) - 1
    df['Dist_EMA50'] = df['Close'] / df.ta.ema(length=50) - 1
    df['RSI_14'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['Volume_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    adx_df = df.ta.adx(length=14)
    atr = df.ta.atr(length=14)
    
    # CORRECCIÓN: Inyectar explícitamente ADX y ATR al DataFrame
    df['ADX_14'] = adx_df['ADX_14']
    df['ATR_14'] = atr['ATR_14'] if isinstance(atr, pd.DataFrame) else atr
    
    # CORRECCIÓN: Contexto HTF (Higher Timeframe - D1) para paridad exacta con XGBoost
    tf_suffix = state.get("timeframe_suffix", "1h")
    htf_tf = mt5.TIMEFRAME_W1 if tf_suffix == "1d" else mt5.TIMEFRAME_D1
    velas_htf = mt5.copy_rates_from_pos(symbol_mt5, htf_tf, 0, 100)
    
    if velas_htf is not None and len(velas_htf) > 10:
        htf = pd.DataFrame(velas_htf)
        htf['time'] = pd.to_datetime(htf['time'], unit='s')
        htf['EMA50_HTF'] = htf['close'].ewm(span=50, adjust=False).mean()
        htf['HTF_Trend'] = np.where(htf['close'] > htf['EMA50_HTF'], 1.0, -1.0)
        htf['HTF_EMA_Slope'] = htf['EMA50_HTF'].pct_change(5)
        
        df_reset = df.reset_index().sort_values('time')
        merged = pd.merge_asof(df_reset, htf[['time', 'HTF_Trend', 'HTF_EMA_Slope']], on='time', direction='backward')
        merged.set_index('time', inplace=True)
        df = merged
    else:
        print(f"   ❌ [Technical Analyst] ERROR: Datos HTF insuficientes. Abortando análisis para no inyectar sesgo.")
        return state # Retorna sin actualizar technical_indicators, forzando HOLD en los siguientes nodos.

    LOOKBACK_ESTRUCTURAL = 15
    df['Max_Estructural'] = df['High'].rolling(window=LOOKBACK_ESTRUCTURAL).max()
    df['Min_Estructural'] = df['Low'].rolling(window=LOOKBACK_ESTRUCTURAL).min()
    
    # Limpiar NaNs recientes para evitar colapsos en la inferencia
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)

    rsi_val = float(df['RSI_14'].iloc[-1])
    adx_val = float(df['ADX_14'].iloc[-1])
    atr_val = float(df['ATR_14'].iloc[-1])
    vol_val = float(df['Vol_20'].iloc[-1])

    print(f"   📈 [Technical Analyst] RSI: {rsi_val:.2f} | ADX: {adx_val:.2f} | ATR: {atr_val:.4f} | Volatilidad (20): {vol_val:.4f}")

    state["technical_indicators"] = {
        "RSI_14": rsi_val, 
        "ATR_14": atr_val, 
        "ADX_14": adx_val,
        "Current_Vol": vol_val,
        "Max_Estructural": float(df['Max_Estructural'].iloc[-1]),
        "Min_Estructural": float(df['Min_Estructural'].iloc[-1])
    }
    state["historical_data"] = df
    return state

# 3. Agente 3: Quant ML Agent (Actualizado a GMM + XGBoost Trinario)
def quant_ml_agent(state: TradingState):
    if "historical_data" not in state or "technical_indicators" not in state: 
        return {"ml_prediction": "HOLD", "ml_confidence": 0.0}
        
    asset = state["asset"]
    tf_suffix = state.get("timeframe_suffix", "1h")
    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    
    model_path = BASE_DIR / f'quant_model_{tf_suffix}_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_{tf_suffix}_{safe_name}.joblib'
    gmm_path = BASE_DIR / f'gmm_regime_{tf_suffix}_{safe_name}.joblib'
    
    prediction, confidence = "HOLD", 0.0
    drift_alert = False
    
    if not model_path.exists() or not gmm_path.exists():
        print(f"   ⚠️ ML Agent: Modelos no disponibles para {asset}. Modo HOLD.")
        return {"ml_prediction": prediction, "ml_confidence": confidence}

    try:
        # Cargar Modelos
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        gmm_data = joblib.load(gmm_path)
        
        gmm = gmm_data['gmm']
        active_regime = gmm_data['active_regime']
        gmm_features = gmm_data['gmm_features']
        
        df_live = state["historical_data"]
        
        # 1. EVALUAR RÉGIMEN ACTUAL (GMM)
        latest_gmm = df_live[gmm_features].iloc[-1:]
        if not latest_gmm.isnull().values.any():
            current_regime = gmm.predict(latest_gmm)[0]
            
            print(f"   🧠 [Quant ML] Régimen GMM detectado: {'Óptimo/Tendencial' if current_regime == active_regime else 'Consolidación/Ruido'}")
            
            # Si el mercado está en el régimen óptimo de tendencia, disparamos XGBoost
            if current_regime == active_regime:
                latest_features = df_live[features].iloc[-1:]
                
                if not latest_features.isnull().values.any():
                    # predict_proba retorna un array de 3 elementos: [Prob_Sell, Prob_Hold, Prob_Buy]
                    probs = model.predict_proba(latest_features)[0]
                    
                    prob_sell = probs[0]
                    prob_buy = probs[2]
                    
                    thresh_buy = gmm_data.get('threshold_buy', 0.50)
                    thresh_sell = gmm_data.get('threshold_sell', 0.50)
                    
                    print(f"   🧠 [Quant ML] Probabilidades XGBoost -> BUY: {prob_buy:.1%} (Thresh: {thresh_buy:.1%}) | SELL: {prob_sell:.1%} (Thresh: {thresh_sell:.1%})")
                    
                    # --- NUEVO: Monitoreo de drift de probabilidades (PSI) ---
                    # Se actualiza el historial en vivo SIEMPRE que tengamos probabilidades válidas
                    # (no solo cuando se dispara una señal), para tener una muestra representativa
                    # de la distribución reciente y compararla contra el baseline del holdout guardado
                    # por train_model_.py. Si el bundle es antiguo (sin 'prob_baseline'), se omite
                    # silenciosamente sin afectar el resto del pipeline.
                    prob_baseline = gmm_data.get('prob_baseline')
                    psi_score = check_probability_drift(safe_name, tf_suffix, prob_buy, prob_sell, prob_baseline)
                    if psi_score is not None:
                        if psi_score > PSI_VETO_THRESHOLD:
                            drift_alert = True
                            print(f"   🚨 [Quant ML] DRIFT SIGNIFICATIVO detectado (PSI={psi_score:.3f} > {PSI_VETO_THRESHOLD}). "
                                  f"El modelo predice fuera de su rango de calibración original. Señal vetada por precaución.")
                        elif psi_score > PSI_WARNING_THRESHOLD:
                            print(f"   ⚠️ [Quant ML] Drift moderado (PSI={psi_score:.3f}). Se sigue operando, pero conviene "
                                  f"programar un re-entrenamiento pronto (FORCE_RETRAIN=1 en train_model_.py).")

                    if drift_alert:
                        pass  # prediction permanece en "HOLD"
                    elif prob_buy > thresh_buy:
                            prediction = "BUY"
                            confidence = prob_buy
                    elif prob_sell > thresh_sell and STRATEGY_TYPE == "LONG_SHORT":
                        # CORRECCIÓN: la condición original 'STRATEGY_TYPE == STRATEGY_TYPE' era una
                        # tautología (siempre True) y por lo tanto el modo LONG_ONLY nunca bloqueaba
                        # señales SELL. Ahora sí respeta la configuración.
                        prediction = "SELL"
                        confidence = prob_sell
                        
            else:
                print("   💤 [Quant ML] Veto algorítmico activado. El mercado carece de la inercia estadística requerida (Régimen 0).")
                
    except Exception as e:
        print(f"   ❌ Error en Inferencia Cuantitativa: {e}")

    return {"ml_prediction": prediction, "ml_confidence": confidence, "drift_alert": drift_alert}

# 4. Agente 4: Fundamental Agent
def fundamental_analyst_agent(state: TradingState):
    if state.get("ml_prediction", "HOLD") == "HOLD":
        print("   🌍 [Fundamental Analyst] Omitido. No hay señal direccional del Agente Quant.") 
        return {"fundamental_sentiment": "NEUTRAL"}
        
    asset = state["asset"]
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        prompt = f"Activo: {asset}. Analiza su contexto macroeconómico actual de forma ultra resumida. Responde estrictamente con una palabra: BULLISH, BEARISH o NEUTRAL."
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        sentiment = resp if resp in ["BULLISH", "BEARISH"] else "NEUTRAL"
        print(f"   🌍 [Fundamental Analyst] Contexto macroeconómico procesado: {sentiment}")
    except:
        sentiment = "NEUTRAL"
        print("   🌍 [Fundamental Analyst] Error en conexión LLM. Asumiendo NEUTRAL.")
    return {"fundamental_sentiment": sentiment}

# 5. Agente 5: Portfolio Manager
def portfolio_manager_agent(state: TradingState):
    ml_signal = state.get("ml_prediction", "HOLD")
    sentiment = state.get("fundamental_sentiment", "NEUTRAL")
    tech = state.get("technical_indicators", {})
    rsi = tech.get("RSI_14", 50.0)
    adx = tech.get("ADX_14", 0.0)
    
    if ml_signal == "HOLD": return {"final_signal": "HOLD"}
    
    if adx < 15: 
        print("   🛑 VETO: Régimen estocástico/lateral (ADX < 15). Carece de inercia.")
        return {"final_signal": "HOLD"}
        
    if (ml_signal == "BUY" and rsi > 75) or (ml_signal == "SELL" and rsi < 25):
        print("   🛑 VETO: Señal estadísticamente sobre-extendida.")
        return {"final_signal": "HOLD"}

    if (ml_signal == "BUY" and sentiment == "BEARISH") or (ml_signal == "SELL" and sentiment == "BULLISH"):
        print("   ⚠️ VETO MACRO: Conflicto frontal entre algoritmo y macroeconomía.")
        return {"final_signal": "HOLD"}

    print(f"   ⚖️ [Portfolio Manager] Señal '{ml_signal}' auditada y APROBADA. Transfiriendo a Gestión de Riesgos.")
    return {"final_signal": ml_signal}

# 6. Agente 6: Risk Manager
def risk_manager_agent(state: TradingState):
    final_signal = state.get("final_signal", "HOLD")
    if final_signal == "HOLD": return {"risk_params": {}}
    
    asset = state["asset"]
    tf_suffix = state.get("timeframe_suffix", "1h")
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    tech = state.get("technical_indicators", {})
    atr = tech.get("ATR_14", 0.0)
    volatility = tech.get("Current_Vol", 0.01)
    current_price = state.get("current_price")
    
    max_estrucl = tech.get("Max_Estructural", current_price)
    min_estrucl = tech.get("Min_Estructural", current_price)
    
    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(symbol_mt5)
    if not account_info or not symbol_info: 
        print("   ❌ [Risk Manager] Error de conexión con terminal MT5 para extraer balance.")
        return {"risk_params": {}}
    
    balance = account_info.balance 
    
    vol_scalar = 0.01 / max(volatility, 0.001) 
    riesgo_operacion_tentativo = (balance * MAX_GLOBAL_PORTFOLIO_RISK_PCT) * min(vol_scalar, 1.0)

    # --- Riesgo cuadrático de portafolio (Var = w' · Corr · w) ---
    # Agrega el riesgo en USD de cada posición abierta (signado por dirección) junto con el de
    # la operación candidata, y resuelve el máximo tamaño permitido para que la RAÍZ de la
    # varianza combinada no supere MAX_PORTFOLIO_CORRELATED_RISK_PCT del balance.
    correlation_scale = 1.0
    corr_matrix = get_correlation_matrix(tf_suffix)
    if corr_matrix is not None and asset in corr_matrix.columns:
        existing_risks = {}
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if pos.magic != 202601:
                    continue
                
                other_asset = REVERSE_ASSET_MAPPING.get(pos.symbol)
                if other_asset is None or other_asset == asset or other_asset not in corr_matrix.columns:
                    continue
                pos_risk_usd = get_position_risk_usd(pos)
                if pos_risk_usd <= 0:
                    continue
                pos_direction = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
                existing_risks[other_asset] = existing_risks.get(other_asset, 0.0) + pos_direction * pos_risk_usd

        candidate_direction = 1 if final_signal == "BUY" else -1
        max_portfolio_risk_usd = balance * MAX_PORTFOLIO_CORRELATED_RISK_PCT
        x_max, existing_variance_A, es_factible = solve_max_correlated_risk(
            asset, candidate_direction, existing_risks, corr_matrix, max_portfolio_risk_usd
        )

        print(f"      ↳ Riesgo cuadrático ya comprometido: ${np.sqrt(max(existing_variance_A, 0.0)):.2f} "
              f"de ${max_portfolio_risk_usd:.2f} permitidos")

        if not es_factible or x_max <= 0:
            print(f"   🛑 [Risk Manager] VETO POR RIESGO CUADRÁTICO DE PORTAFOLIO: las posiciones abiertas "
                  f"correlacionadas ya agotan (o superan) el presupuesto de ${max_portfolio_risk_usd:.2f}.")
            return {"risk_params": {}}
        elif x_max < riesgo_operacion_tentativo:
            correlation_scale = x_max / riesgo_operacion_tentativo
            print(f"   ⚠️ [Risk Manager] Riesgo cuadrático de portafolio alto. Reduciendo esta operación "
                  f"x{correlation_scale:.2f} (máximo permitido: ${x_max:.2f}).")

    riesgo_operacion_usd = riesgo_operacion_tentativo * correlation_scale
    
    colchon_ruido = atr * 0.25 
    
    if final_signal == "BUY":
        stop_loss_absoluto = min_estrucl - colchon_ruido
        sl_distance = max(current_price - stop_loss_absoluto, atr * 1.5)
    else:
        stop_loss_absoluto = max_estrucl + colchon_ruido
        sl_distance = max(stop_loss_absoluto - current_price, atr * 1.5)
        
    confianza = state.get("ml_confidence", 0.0)
    ratio_riesgo_beneficio = 2.0 if confianza > 0.65 else 1.5
    tp_distance = sl_distance * ratio_riesgo_beneficio
    
    ticks_at_risk = sl_distance / symbol_info.trade_tick_size
    tick_val = symbol_info.trade_tick_value
    
    raw_lot_size = riesgo_operacion_usd / (ticks_at_risk * tick_val) if ticks_at_risk > 0 else 0
    lot_steps = int(raw_lot_size / symbol_info.volume_step)
    lot_size = lot_steps * symbol_info.volume_step
    
    if lot_size < symbol_info.volume_min:
        print(f"   🛑 [Risk Manager] VETO DE RIESGO: Lote calculado ({lot_size}) es inferior al mínimo permitido por el broker.")
        return {"risk_params": {}}
        
    lot_size = min(lot_size, symbol_info.volume_max)
    step_decimals = len(str(symbol_info.volume_step).split('.')[1]) if '.' in str(symbol_info.volume_step) else 0
    lot_size = round(lot_size, step_decimals)
    
    if final_signal == "BUY":
        sl_price = current_price - sl_distance
        tp_price = current_price + tp_distance
    else:
        sl_price = current_price + sl_distance
        tp_price = current_price - tp_distance
    
    print(f"   🛡️ [Risk Manager] Diseño Estructural Completado:")
    print(f"      - Capital Arriesgado: ${riesgo_operacion_usd:.2f} ({(riesgo_operacion_usd/balance)*100:.2f}% del balance)")
    print(f"      - Exposición (Lotes): {lot_size}")
    print(f"      - Niveles Estimados -> ENTRADA: {current_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
    
    return {"risk_params": {
        "lot_size": lot_size,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance
    }}

# 7. Agente 7: Execution Agent
def execution_agent(state: TradingState):
    signal = state.get("final_signal", "HOLD")
    params = state.get("risk_params", {})
    if signal == "HOLD" or not params: return {"final_execution": "   ⏸️ [Execution Agent] Pipeline cerrado sin apertura de posiciones (HOLD)."}
    
    asset = state["asset"]
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    tick = mt5.symbol_info_tick(symbol_mt5)
    
    lot_size = params["lot_size"]
    sl_dist = params["sl_distance"]
    tp_dist = params["tp_distance"]
    
    if signal == "BUY":
        price = tick.ask
        sl = price - sl_dist
        tp = price + tp_dist
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = price + sl_dist
        tp = price - tp_dist
        order_type = mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 10,
        "magic": 202601,
        "comment": "Alpha Agent",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        msg = f"❌ [Execution Agent] ERROR DE BROKER: {result.comment} ({result.retcode})"
    else:
        msg = f"✅ [Execution Agent] ÉXITO | Ticket: {result.order} | Activo: {symbol_mt5} | {signal} a {price} | Lotes: {lot_size}"
        log_trade_execution(asset, signal, state.get("ml_confidence", 0.0), signal, result.order, result.price, lot_size)
        
    return {"final_execution": msg}

# Ensamblaje 
workflow = StateGraph(TradingState)
for node in [("market_data", market_data_agent), ("technical_analyst", technical_analyst_agent), 
             ("quant_ml", quant_ml_agent), ("fundamental_analyst", fundamental_analyst_agent), 
             ("portfolio_manager", portfolio_manager_agent), ("risk_manager", risk_manager_agent), 
             ("execution", execution_agent)]: workflow.add_node(node[0], node[1])

workflow.add_edge("market_data", "technical_analyst")
workflow.add_edge("technical_analyst", "quant_ml")
workflow.add_edge("quant_ml", "fundamental_analyst")
workflow.add_edge("fundamental_analyst", "portfolio_manager")
workflow.add_edge("portfolio_manager", "risk_manager")
workflow.add_edge("risk_manager", "execution")
workflow.add_edge("execution", END)
workflow.set_entry_point("market_data")
app = workflow.compile()

if __name__ == "__main__":
    print("\n🚀 INICIANDO ECOSISTEMA DE AGENTES AUTÓNOMOS")
    path = os.getenv("MT5_PATH")
    if mt5.initialize(path=path) and mt5.login(login=int(os.getenv("MT5_LOGIN")), password=os.getenv("MT5_PASSWORD"), server=os.getenv("MT5_SERVER")):
        for asset, asset_mt5 in ASSET_MAPPING.items():
            print(f"\n--- INFERENCIA: {asset} ---")
            if is_market_open(asset):
               try: 
                    # Invocación secuencial para aprovechar modelos 1D y 1H guardados ---
                    for tf_eval in ["1d", "1h"]:
                        print(f"\n   ⏱️ [Orquestador] Procesando ventana temporal: {tf_eval.upper()}")
                        resultado = app.invoke({"asset": asset, "timeframe_suffix": tf_eval})
                        print(resultado["final_execution"])
               except Exception as e: 
                    print(f"   ❌ Fallo crítico en {asset}: {e}")
        mt5.shutdown()