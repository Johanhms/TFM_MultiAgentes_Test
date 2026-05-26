import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
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

def train_xgboost_for_asset(asset: str):
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO XGBOOST PARA: {asset} ({symbol_mt5})")
    print(f"{'='*60}")
    
    print("📥 1. Descargando datos masivos (10,000 velas en 1H) desde MT5...")
    # Obtenemos las últimas 10,000 velas del servidor del bróker
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 10000)
    
    if velas_mt5 is None or len(velas_mt5) < 200:
        print(f"❌ ERROR: Datos insuficientes en MT5 para {symbol_mt5}. Saltando...")
        return

    # Transformación del formato de MT5 al formato estándar que Pandas-TA requiere
    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'tick_volume': 'Volume'
    }, inplace=True)
    df.set_index('time', inplace=True)
    
    # Filtramos solo las columnas necesarias
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("⚙️ 2. Feature Engineering...")
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    roc = df.ta.roc(length=10)

    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1

    macd_cols = macd.columns.tolist()
    rsi_name = rsi.name
    atr_name = atr.name
    roc_name = roc.name

    df = pd.concat([df, macd, rsi, atr, roc], axis=1)
    features = macd_cols + [rsi_name, atr_name, roc_name, 'Retorno_1H', 'Volatilidad_10H', 'Distancia_SMA20']
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']

    # Separación Out-of-Sample (10% final para la prueba de fuego)
    test_size = int(len(df) * 0.1)
    X_train_cv = X.iloc[:-test_size]
    y_train_cv = y.iloc[:-test_size]
    X_test_final = X.iloc[-test_size:]
    y_test_final = y.iloc[-test_size:]

    print("🧠 3. Entrenando XGBoost con TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parámetros optimizados para Gradient Boosting
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
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base, 
        param_distributions=param_dist, 
        n_iter=20,               
        cv=tscv, 
        scoring='accuracy', 
        random_state=42,
        n_jobs=-1
    )
    
    print("⏳ Explorando 20 arquitecturas aleatorias... ")
    random_search.fit(X_train_cv, y_train_cv)
    best_model = random_search.best_estimator_
    
    print(f"   🎯 Mejores parámetros encontrados: {random_search.best_params_}")

    print("📊 4. Validación Científica (Out-of-Sample)...")
    y_pred = best_model.predict(X_test_final)
    
    acc = accuracy_score(y_test_final, y_pred) * 100
    report = classification_report(y_test_final, y_pred, target_names=['BAJA (0)', 'SUBE (1)'])
    
    print(f"\n   📈 Accuracy General: {acc:.2f}%")
    print("   📋 Reporte de Clasificación:")
    print(report)

    # Solo guardamos el modelo si supera el umbral del 51% (evitar guardar modelos basura)
    if acc > 51.0:
        print("💾 5. Modelo aprobado. Guardando en .joblib...")
        safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
        
        model_path = BASE_DIR / f'quant_model_1h_{safe_name}.joblib'
        features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'
        
        joblib.dump(best_model, model_path)
        joblib.dump(features, features_path)
        print(f"   ✅ Guardado: {model_path.name}")
    else:
        print("   ⚠️ EL MODELO ES DEFICIENTE (<51%). No se guardará para proteger el capital.")

if __name__ == "__main__":
    # Inicializamos la conexión global a MT5 al arrancar el script
    if not mt5.initialize():
        print("❌ ERROR CRÍTICO: MetaTrader 5 no está abierto o no se pudo inicializar.")
    else:
        for asset in ASSETS_TO_TRAIN:
            train_xgboost_for_asset(asset)
        
        # Cerramos la conexión educadamente al terminar
        mt5.shutdown()
        print("\n✅ ENTRENAMIENTO XGBOOST FINALIZADO.")