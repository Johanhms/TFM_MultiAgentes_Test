import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ASSETS_TO_TRAIN = ["BTC-USD", "EURUSD=X", "GBPUSD=X", "GC=F", "^GSPC", "CL=F", "^DJI", "NVDA"]
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

# Hiperparámetro Clave: ¿Cuántas horas atrás debe "recordar" la red para predecir la siguiente?
LOOKBACK_WINDOW = 24 

def create_sequences(data, labels, lookback):
    """Transforma datos 2D tabulares en Tensores 3D (Muestras, Pasos de Tiempo, Variables)"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

def train_lstm_for_asset(asset: str):
    print(f"\n{'='*60}")
    print(f"🧠 INICIANDO ENTRENAMIENTO DEEP LEARNING (LSTM) PARA: {asset}")
    print(f"{'='*60}")
    
    # 1. Extracción y Feature Engineering (Igual que antes)
    df = yf.Ticker(asset).history(period="730d", interval="1h")
    if df.empty or len(df) < 200: return

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    
    df = pd.concat([df, macd, rsi, atr], axis=1)
    features = macd.columns.tolist() + [rsi.name, atr.name, 'Retorno_1H', 'Volatilidad_10H']
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # 2. ESCALADO (Obligatorio en Redes Neuronales)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[features])
    labels = df['Target'].values

    # 3. CREACIÓN DE SECUENCIAS 3D
    X, y = create_sequences(scaled_features, labels, LOOKBACK_WINDOW)

    # Separación Train/Test (10% final fuera de la muestra)
    test_size = int(len(X) * 0.1)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # 4. ARQUITECTURA DE LA RED NEURONAL (LSTM)
    print("⚙️ Construyendo y entrenando la Red Neuronal...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3), # Previene sobreajuste apagando 30% de las neuronas
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid') # Salida Binaria: Probabilidad entre 0 y 1
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping: Detiene el entrenamiento si la red empieza a sobreajustarse a los datos de validación
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Entrenamiento
    model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[early_stop],
        verbose=0 # Ponlo en 1 si quieres ver la barra de progreso por cada época
    )

    # 5. EVALUACIÓN Y GUARDADO
    print("📊 Validación Científica (Out-of-Sample)...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int) # Umbral del 50%
    
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n   📈 Accuracy DL: {acc:.2f}%")
    print("   📋 Reporte:")
    print(classification_report(y_test, y_pred, target_names=['BAJA', 'SUBE']))

    if acc > 51.0:
        safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
        
        # En DL guardamos la red (.keras) y el escalador (.joblib)
        model_path = BASE_DIR / f'dl_model_1h_{safe_name}.keras'
        scaler_path = BASE_DIR / f'dl_scaler_1h_{safe_name}.joblib'
        features_path = BASE_DIR / f'model_features_1h_{safe_name}.joblib'
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(features, features_path)
        print(f"   ✅ Ecosistema DL guardado exitosamente.")
    else:
        print("   ⚠️ EL MODELO ES DEFICIENTE. Descartado.")

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Oculta warnings de TensorFlow
    
    for asset in ASSETS_TO_TRAIN:
        train_lstm_for_asset(asset)
    print("\n✅ ENTRENAMIENTO DEEP LEARNING FINALIZADO.")