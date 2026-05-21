import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
import warnings

# Ignoramos warnings de yfinance para una consola más limpia
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

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def train_xgboost_for_asset(asset: str):
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO XGBOOST PARA: {asset}")
    print(f"{'='*60}")
    
    print("📥 1. Descargando datos (730 días en 1H)...")
    df = yf.Ticker(asset).history(period="730d", interval="1h")
    
    if df.empty or len(df) < 200:
        print(f"❌ ERROR: Datos insuficientes para {asset}. Saltando...")
        return

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
        'n_estimators': [100, 200, 300],            # Reducimos el límite máximo
        'learning_rate': [0.01, 0.05, 0.1],         # Eliminamos el 0.2 (demasiado agresivo)
        'max_depth': [2, 3, 4, 5],                  # CLAVE: Árboles muy poco profundos
        'subsample': [0.5, 0.6, 0.7, 0.8],          # CLAVE: Nunca usamos el 100% de las filas
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],   # CLAVE: Nunca usamos el 100% de las columnas
        'gamma': [0, 0.1, 0.5, 1]                   # NUEVO: Penalización por crear ramas complejas
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
        n_iter=20,               # <--- Puedes subir este número si tienes más tiempo
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
    
    # --- EL NUEVO BLOQUE DE AUDITORÍA ---
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
    for asset in ASSETS_TO_TRAIN:
        train_xgboost_for_asset(asset)
    print("\n✅ ENTRENAMIENTO XGBOOST FINALIZADO.")