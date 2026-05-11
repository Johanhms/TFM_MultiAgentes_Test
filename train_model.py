import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("📥 1. Descargando datos históricos (5 años para CV)...")
df = yf.Ticker("BTC-USD").history(period="730d", interval="1h")

print("⚙️ 2. Feature Engineering Profundo...")
# 2.1 Indicadores Base
macd = df.ta.macd(fast=12, slow=26, signal=9)
rsi = df.ta.rsi(length=14)
atr = df.ta.atr(length=14)
roc = df.ta.roc(length=10)

# 2.2 NUEVO: Features de Momentum y Volatilidad (La clave para mejorar el Accuracy)
df['Retorno_1H'] = df['Close'].pct_change()
df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1

# Extraemos nombres dinámicamente
macd_cols = macd.columns.tolist()
rsi_name = rsi.name
atr_name = atr.name
roc_name = roc.name

# Unimos todo
df = pd.concat([df, macd, rsi, atr, roc], axis=1)

# Lista final de variables
features = macd_cols + [rsi_name, atr_name, roc_name, 'Retorno_1H', 'Volatilidad_10H', 'Distancia_SMA20']
print(f"   Total de variables analizadas: {len(features)}")

# --- CREACIÓN DEL TARGET ---
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

X = df[features]
y = df['Target']

# Separamos el 10% final de los datos para la prueba FINAL a ciegas
test_size = int(len(df) * 0.1)
X_train_cv = X.iloc[:-test_size]
y_train_cv = y.iloc[:-test_size]
X_test_final = X.iloc[-test_size:]
y_test_final = y.iloc[-test_size:]

print("🧠 3. Configurando Validación Cruzada (TimeSeriesSplit)...")
# Dividimos la historia en 5 bloques secuenciales (Walk-Forward)
tscv = TimeSeriesSplit(n_splits=5)

# Definimos la grilla de parámetros a probar
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20],
    'max_features': ['sqrt', 'log2']
}

# Iniciamos el modelo base
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# GridSearch buscará la mejor combinación usando nuestra validación cruzada temporal
gsearch = GridSearchCV(
    estimator=rf_base, 
    cv=tscv, 
    param_grid=param_grid, 
    scoring='accuracy', 
    n_jobs=-1 # Usa todos los núcleos de tu procesador
)

print("⏳ Buscando patrones horarios... (Esto tomará unos segundos)")
gsearch.fit(X_train_cv, y_train_cv)

best_model = gsearch.best_estimator_
print(f"   Mejores hiperparámetros: {gsearch.best_params_}")

print("📊 4. Evaluación Final (Out-of-Sample)...")
# Probamos el mejor modelo contra ese 10% de datos que NUNCA vio ni en la validación
y_pred = best_model.predict(X_test_final)
acc = accuracy_score(y_test_final, y_pred) * 100
print(f"Precisión (Accuracy) Real Esperada: {acc:.2f}%")

print("💾 5. Guardando el ecosistema de ML...")
joblib.dump(best_model, 'quant_model_1h.joblib')
joblib.dump(features, 'model_features_1h.joblib')
print("✅ Proceso finalizado.")