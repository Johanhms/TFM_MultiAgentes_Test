import os
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from pathlib import Path
from dotenv import load_dotenv
import warnings
import numpy as np
import optuna

# Ignoramos warnings y silenciamos los logs excesivos de Optuna para una consola limpia
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ASSETS_TO_TRAIN = [
    "BTC-USD", "EURUSD=X", "GBPUSD=X", "GC=F", 
    "^GSPC", "CL=F", "^DJI", "NVDA"
]

ASSET_MAPPING = {
    "BTC-USD": "BTCUSD", "EURUSD=X": "EURUSD.sml", "GBPUSD=X": "GBPUSD.sml",
    "GC=F": "XAUUSD.sml", "^GSPC": "US500", "CL=F": "USOIL.sml", 
    "^DJI": "US30", "NVDA": "NVDA_CFD.US"
}

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def calcular_metricas_financieras(y_pred, df_test, timeframe, retorno_col):
    """
    Simulación financiera indexada y adaptada dinámicamente al marco temporal (H1 o D1).
    """
    señales = np.where(y_pred == 1, 1, -1)
    retornos_futuros = df_test[retorno_col].shift(-1)
    retornos_estrategia = señales * retornos_futuros
    retornos_estrategia = retornos_estrategia.dropna()
    
    if len(retornos_estrategia) == 0: return 0.0, 0.0, 0.0

    # AJUSTE QUANT: Factor de anualización asimétrico según el Timeframe
    if timeframe == mt5.TIMEFRAME_H1:
        factor_anualizacion = np.sqrt(252 * 24) # Horas operables año
    else:
        factor_anualizacion = np.sqrt(252)      # Días operables año

    media_retornos = retornos_estrategia.mean()
    volatilidad_retornos = retornos_estrategia.std()
    sharpe_ratio = (media_retornos / volatilidad_retornos) * factor_anualizacion if volatilidad_retornos > 0 else 0.0

    ganancias_brutas = retornos_estrategia[retornos_estrategia > 0].sum()
    perdidas_brutas = np.abs(retornos_estrategia[retornos_estrategia < 0].sum())
    profit_factor = ganancias_brutas / perdidas_brutas if perdidas_brutas > 0 else 0.0

    curva_equidad = (1 + retornos_estrategia).cumprod()
    picos_acumulados = curva_equidad.cummax()
    drawdowns = (curva_equidad - picos_acumulados) / picos_acumulados
    max_drawdown = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0

    return sharpe_ratio, profit_factor, max_drawdown

def train_xgboost_for_asset(asset: str, timeframe):
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    # Configuramos los parámetros visuales e históricos según el Timeframe
    tf_label = "D1_MACRO" if timeframe == mt5.TIMEFRAME_D1 else "H1_MICRO"
    suffix = "1d" if timeframe == mt5.TIMEFRAME_D1 else "1h"
    n_velas = 1500 if timeframe == mt5.TIMEFRAME_D1 else 10000 # Muestreo asimétrico inteligente
    
    retorno_col = f'Retorno_{suffix.upper()}'
    volatilidad_col = f'Volatilidad_10{suffix.upper()}'
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO PIPELINE XGBOOST (OPTUNA) [{tf_label}] PARA: {asset}")
    print(f"{'='*60}")
    
    print(f"📥 1. Descargando datos masivos ({n_velas} velas en {suffix.upper()}) desde MT5...")
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, timeframe, 0, n_velas)
    
    if velas_mt5 is None or len(velas_mt5) < 300:
        print(f"❌ ERROR: Datos insuficientes en MT5 para {symbol_mt5} ({tf_label}). Saltando...")
        return

    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("⚙️ 2. Feature Engineering Sincronizado...")
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    rsi = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    roc = df.ta.roc(length=10)
    adx_df = df.ta.adx(length=14)
    
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_50_Slope'] = df['EMA_50'].diff(periods=3)
    
    obv = df.ta.obv()
    df['OBV'] = obv
    df['OBV_Slope'] = df['OBV'].diff(periods=3)

    # Marcaje adaptativo de columnas
    df[retorno_col] = df['Close'].pct_change()
    df[volatilidad_col] = df[retorno_col].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1

    macd_cols = macd.columns.tolist()
    adx_cols = ['ADX_14'] if adx_df is not None else []
    
    df = pd.concat([df, macd, rsi, atr, roc, adx_df], axis=1)
    
    features = macd_cols + adx_cols + [
        rsi.name, atr.name, roc.name, 
        'EMA_20', 'EMA_50', 'EMA_50_Slope', 'OBV', 'OBV_Slope', 
        retorno_col, volatilidad_col, 'Distancia_SMA20'
    ]
    
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']

    test_size = int(len(df) * 0.1)
    X_train = X.iloc[:-test_size]
    y_train = y.iloc[:-test_size]
    X_test_final = X.iloc[-test_size:]
    y_test_final = y.iloc[-test_size:]

    print("🧠 3. Optimizando Hiperparámetros con Inteligencia Bayesiana (Optuna)...")
    
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 7),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0)
        }

        tscv = TimeSeriesSplit(n_splits=3)
        auc_scores = []

        for train_index, valid_index in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

            model = xgb.XGBClassifier(**param)
            model.fit(X_tr, y_tr, verbose=False)
            
            preds_proba = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, preds_proba)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    print(f"   🎯 Optuna finalizado. Mejores parámetros: {study.best_params}")

    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    print("📊 4. Validación Científica Out-of-Sample...")
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
    
    acc = accuracy_score(y_test_final, y_pred) * 100
    auc = roc_auc_score(y_test_final, y_pred_proba) * 100
    
    sharpe, profit_factor, max_dd = calcular_metricas_financieras(y_pred, df.iloc[-test_size:], timeframe, retorno_col)
    
    print(f"\n   📈 MÉTRICAS DE LABORATORIO (ML):")
    print(f"      Accuracy General: {acc:.2f}%")
    print(f"      ROC-AUC Score:  {auc:.2f}%")
    
    print(f"\n   💰 MÉTRICAS FINANCIERAS (Simulación Out-of-Sample):")
    print(f"      Sharpe Ratio:    {sharpe:.2f}")
    print(f"      Profit Factor:   {profit_factor:.2f}")
    print(f"      Max Drawdown:    {max_dd:.2f}%")

    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    
    # GUARDADO ASIMÉTRICO: Los nombres de los archivos reflejan su horizonte temporal
    model_path = BASE_DIR / f'quant_model_{suffix}_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_{suffix}_{safe_name}.joblib'

    # Ajustamos un filtro un poco más flexible para el modelo Macro D1, ya que capturar tendencias diarias es más exigente
    umbral_pf = 1.01 if timeframe == mt5.TIMEFRAME_D1 else 1.02

    if acc > 50.5 and auc > 50.5 and profit_factor >= umbral_pf:
        print(f"\n💾 5. Modelo [{tf_label}] aprobado. Guardando en producción...")
        joblib.dump(best_model, model_path)
        joblib.dump(features, features_path)
        print(f"   ✅ Archivo serializado con éxito: {model_path.name}")
    else:
        print(f"\n   ⚠️ ALERTA: El modelo nuevo [{tf_label}] no supera los criterios de calidad.")
        
        if model_path.exists() and features_path.exists():
            print(f"   🛡️ POLÍTICA FALLBACK: Se mantiene el modelo anterior [{suffix}] operativo.")
        else:
            print(f"   ❌ AVISO: No existe un modelo previo. {asset} ({suffix.upper()}) no operará.")

if __name__ == "__main__":
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
            print(f"❌ ERROR CRÍTICO: Fallo de login en MT5. Revisa tus credenciales.")
        else:
            # EL DOCTORADO DEL PIPELINE: El bucle entrena el horizonte Macro y luego el Micro
            for asset in ASSETS_TO_TRAIN:
                train_xgboost_for_asset(asset, mt5.TIMEFRAME_D1) # Entrenar el Estratega Diario
                train_xgboost_for_asset(asset, mt5.TIMEFRAME_H1) # Entrenar el Francotirador Horario
            
        mt5.shutdown()
        print("\n✅ PIPELINE MULTI-TEMPORAL EN ALTA RESOLUCIÓN COMPLETADO.")