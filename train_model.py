import os
import numpy as np
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
from dotenv import load_dotenv
import warnings

# Ignoramos warnings para una consola más limpia
warnings.filterwarnings("ignore")

ASSETS_TO_TRAIN = [
    "BTC-USD", 
    "EURUSD=X", 
    "GBPUSD=X", 
    "GC=F", 
    "^GSPC",
    "CL=F",
    "^DJI",
    "NVDA"
]

# Diccionario de traducción para que MT5 entienda qué descargar
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

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def calcular_metricas_financieras(y_pred, df_test):
    """
    Convierte las predicciones de Machine Learning en una simulación financiera
    corrigiendo el desfase temporal mediante shift(-1) para capturar el retorno futuro.
    """
    # Las señales del modelo se transforman a direcciones de mercado: 1 (Long), -1 (Short)
    señales = np.where(y_pred == 1, 1, -1)
    
    # CORRECCIÓN CRÍTICA: La predicción en 't' evalúa la vela que se desarrollará en 't+1'
    retornos_futuros = df_test['Retorno_1H'].shift(-1)
    retornos_estrategia = señales * retornos_futuros
    
    # Eliminamos el último registro indexado debido al desplazamiento del shift
    retornos_estrategia = retornos_estrategia.dropna()
    
    if len(retornos_estrategia) == 0:
        return 0.0, 0.0, 0.0

    # --- SHARPE RATIO ANUALIZADO ---
    factor_anualizacion = np.sqrt(252 * 24)  # Muestreo horario intradía
    media_retornos = retornos_estrategia.mean()
    volatilidad_retornos = retornos_estrategia.std()
    sharpe_ratio = (media_retornos / volatilidad_retornos) * factor_anualizacion if volatilidad_retornos > 0 else 0.0

    # --- PROFIT FACTOR ---
    ganancias_brutas = retornos_estrategia[retornos_estrategia > 0].sum()
    perdidas_brutas = np.abs(retornos_estrategia[retornos_estrategia < 0].sum())
    profit_factor = ganancias_brutas / perdidas_brutas if perdidas_brutas > 0 else 0.0

    # --- MAXIMUM DRAWDOWN ---
    curva_equidad = (1 + retornos_estrategia).cumprod()
    picos_acumulados = curva_equidad.cummax()
    drawdowns = (curva_equidad - picos_acumulados) / picos_acumulados
    max_drawdown = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0

    return sharpe_ratio, profit_factor, max_drawdown


def train_xgboost_for_asset(asset: str):
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO XGBOOST PARA: {asset} ({symbol_mt5})")
    print(f"{'='*60}")
    
    print("📥 1. Descargando datos masivos (10,000 velas en 1H) desde MT5...")
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 10000)
    
    if velas_mt5 is None or len(velas_mt5) < 300: # Subimos a 300 mínimo para calcular las EMAs largas sin error
        print(f"❌ ERROR: Datos insuficientes en MT5 para {symbol_mt5}. Saltando...")
        return

    # Transformación del formato de MT5 al formato estándar
    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("⚙️ 2. Feature Engineering (Sincronizado con Agentes)...")
    # Osciladores clásicos
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    roc = df.ta.roc(length=10)
    
    # NUEVO: Indicadores de Régimen, Tendencia y Volumen (El "Espejo" del Agente)
    adx_df = df.ta.adx(length=14)
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_50_Slope'] = df['EMA_50'].diff(periods=3)
    
    obv = df.ta.obv()
    df['OBV'] = obv
    df['OBV_Slope'] = df['OBV'].diff(periods=3)

    # Features estadísticas base
    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1

    # Consolidación del DataFrame
    macd_cols = macd.columns.tolist()
    adx_cols = ['ADX_14'] if adx_df is not None else []
    
    df = pd.concat([df, macd, rsi, atr, roc, adx_df], axis=1)
    
    # Declaración ESTRICTA de las features que aprenderá el modelo
    features = macd_cols + adx_cols + [
        rsi.name, atr.name, roc.name, 
        'EMA_20', 'EMA_50', 'EMA_50_Slope', 'OBV', 'OBV_Slope', 
        'Retorno_1H', 'Volatilidad_10H', 'Distancia_SMA20'
    ]
    
    # Definición del Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Limpiamos todos los NaNs generados por los cálculos (ej. EMA_50 requiere 50 velas)
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']

    # Separación Out-of-Sample (10% final) respetando la cronología
    test_size = int(len(df) * 0.1)
    X_train_cv = X.iloc[:-test_size]
    y_train_cv = y.iloc[:-test_size]
    X_test_final = X.iloc[-test_size:]
    y_test_final = y.iloc[-test_size:]

    print("🧠 3. Entrenando XGBoost con TimeSeriesSplit y Optimización ROC-AUC...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_dist = {
        'n_estimators': [100, 200, 300],            
        'learning_rate': [0.01, 0.05, 0.1],         
        'max_depth': [2, 3, 4, 5],                  
        'subsample': [0.5, 0.6, 0.7, 0.8],          
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],   
        'gamma': [0, 0.1, 0.5, 1]                   
    }

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # MEJORA: scoring='roc_auc' en lugar de 'accuracy'
    random_search = RandomizedSearchCV(
        estimator=xgb_base, 
        param_distributions=param_dist, 
        n_iter=20,               
        cv=tscv, 
        scoring='roc_auc', 
        random_state=42,
        n_jobs=-1
    )
    
    print("⏳ Explorando 20 arquitecturas aleatorias... ")
    random_search.fit(X_train_cv, y_train_cv)
    best_model = random_search.best_estimator_
    
    print(f"   🎯 Mejores parámetros encontrados: {random_search.best_params_}")

    print("📊 4. Validación Científica (Out-of-Sample)...")
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
    
    acc = accuracy_score(y_test_final, y_pred) * 100
    auc = roc_auc_score(y_test_final, y_pred_proba) * 100
    report = classification_report(y_test_final, y_pred, target_names=['BAJA (0)', 'SUBE (1)'])
    
    # NUEVO: Evaluamos las métricas financieras en el dataset de prueba
    sharpe, profit_factor, max_dd = calcular_metricas_financieras(y_pred, df.iloc[-test_size:])
    
    print(f"\n   📈 MÉTRICAS DE LABORATORIO (ML):")
    print(f"      Accuracy General: {acc:.2f}%")
    print(f"      ROC-AUC Score:  {auc:.2f}%")
    
    print(f"\n   💰 MÉTRICAS FINANCIERAS (Simulación Out-of-Sample):")
    print(f"      Sharpe Ratio:    {sharpe:.2f}  (Objetivo: > 1.5)")
    print(f"      Profit Factor:   {profit_factor:.2f}  (Objetivo: > 1.2)")
    print(f"      Max Drawdown:    {max_dd:.2f}% (Objetivo: > -15.0%)")
    
    
    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    model_path = BASE_DIR / f'quant_model_1h_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'

    # Ajustamos a un umbral realista de rentabilidad para la fase de pre-filtrado
    if acc > 51.0 and auc > 51.0 and profit_factor >= 1.02:
        print("\n💾 5. Modelo aprobado por Criterio Múltiple. Actualizando entornos de producción...")
        joblib.dump(best_model, model_path)
        joblib.dump(features, features_path)
        print(f"   ✅ Archivo serializado con éxito: {model_path.name}")
    else:
        print("\n   ⚠️ ALERTA: El nuevo modelo no supera los criterios mínimos de calidad.")
        
        # CORRECCIÓN DE SEGURIDAD: Lógica de Fallback Institucional
        if model_path.exists() and features_path.exists():
            print(f"   🛡️ POLÍTICA FALLBACK ACTIVADA: Se mantiene el modelo anterior de {asset} operativo.")
            print("      Evitando paradas de producción. El bot mantendrá su configuración previa.")
        else:
            print(f"   ❌ AVISO: No existe un modelo previo en la carpeta. {asset} no operará esta semana.")

if __name__ == "__main__":
    # NUEVO: Cargamos las variables de entorno para una conexión segura a la cuenta del bróker
    load_dotenv()
    
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")

    if not mt5.initialize(path=path):
        print(f"❌ ERROR CRÍTICO: Fallo al inicializar MT5. Código: {mt5.last_error()}")
    else:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            print(f"❌ ERROR CRÍTICO: Fallo de login en MT5. Revisa tus credenciales en el .env.")
        else:
            for asset in ASSETS_TO_TRAIN:
                train_xgboost_for_asset(asset)
            
        mt5.shutdown()
        print("\n✅ ENTRENAMIENTO XGBOOST FINALIZADO.")