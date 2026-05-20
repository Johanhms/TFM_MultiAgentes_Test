import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# 1. Definimos los activos a entrenar (Debe coincidir con tu portafolio)
ASSETS_TO_TRAIN = [
    "BTC-USD", 
    "EURUSD=X", 
    "GBPUSD=X", 
    "GC=F", 
    "^GSPC"
]

# 2. Configuración de rutas dinámicas para asegurar compatibilidad en cualquier PC
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def train_for_asset(asset: str):
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO ENTRENAMIENTO PARA: {asset}")
    print(f"{'='*50}")
    
    print("📥 1. Descargando datos históricos (730 días en 1H)...")
    df = yf.Ticker(asset).history(period="730d", interval="1h")
    
    if df.empty or len(df) < 200:
        print(f"❌ ERROR: Datos insuficientes para {asset}. Saltando...")
        return

    print("⚙️ 2. Feature Engineering Profundo...")
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

    test_size = int(len(df) * 0.1)
    X_train_cv = X.iloc[:-test_size]
    y_train_cv = y.iloc[:-test_size]
    X_test_final = X.iloc[-test_size:]
    y_test_final = y.iloc[-test_size:]

    print("🧠 3. Configurando Validación Cruzada (Walk-Forward)...")
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [10, 20],
        'max_features': ['sqrt', 'log2']
    }

    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    gsearch = GridSearchCV(estimator=rf_base, cv=tscv, param_grid=param_grid, scoring='accuracy', n_jobs=-1)

    print("⏳ Buscando patrones horarios...")
    gsearch.fit(X_train_cv, y_train_cv)
    best_model = gsearch.best_estimator_
    
    y_pred = best_model.predict(X_test_final)
    acc = accuracy_score(y_test_final, y_pred) * 100
    print(f"📊 Precisión (Accuracy) Out-of-Sample: {acc:.2f}%")

    print("💾 4. Guardando el ecosistema de ML dinámicamente...")
    # Limpiamos el nombre del activo para que el archivo no tenga caracteres problemáticos
    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    
    model_path = BASE_DIR / f'quant_model_1h_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'
    
    joblib.dump(best_model, model_path)
    joblib.dump(features, features_path)
    print(f"   ✅ Guardado: {model_path.name}")

if __name__ == "__main__":
    for asset in ASSETS_TO_TRAIN:
        train_for_asset(asset)
    print("\n✅ ENTRENAMIENTO DE PORTAFOLIO MULTIACTIVO FINALIZADO.")