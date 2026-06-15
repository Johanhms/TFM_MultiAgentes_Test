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

def calcular_metricas_financieras(y_pred, df_test):
    señales = np.where(y_pred == 1, 1, -1)
    retornos_futuros = df_test['Retorno_1H'].shift(-1)
    retornos_estrategia = señales * retornos_futuros
    retornos_estrategia = retornos_estrategia.dropna()
    
    if len(retornos_estrategia) == 0: return 0.0, 0.0, 0.0

    factor_anualizacion = np.sqrt(252 * 24)
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

def train_xgboost_for_asset(asset: str):
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO XGBOOST (OPTUNA) PARA: {asset}")
    print(f"{'='*60}")
    
    print("📥 1. Descargando datos masivos (10,000 velas en 1H) desde MT5...")
    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_H1, 0, 10000)
    
    if velas_mt5 is None or len(velas_mt5) < 300:
        print(f"❌ ERROR: Datos insuficientes en MT5 para {symbol_mt5}. Saltando...")
        return

    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("⚙️ 2. Feature Engineering (Sincronizado con Agentes)...")
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

    df['Retorno_1H'] = df['Close'].pct_change()
    df['Volatilidad_10H'] = df['Retorno_1H'].rolling(window=10).std()
    df['Distancia_SMA20'] = (df['Close'] / df.ta.sma(length=20)) - 1

    macd_cols = macd.columns.tolist()
    adx_cols = ['ADX_14'] if adx_df is not None else []
    
    df = pd.concat([df, macd, rsi, atr, roc, adx_df], axis=1)
    
    features = macd_cols + adx_cols + [
        rsi.name, atr.name, roc.name, 
        'EMA_20', 'EMA_50', 'EMA_50_Slope', 'OBV', 'OBV_Slope', 
        'Retorno_1H', 'Volatilidad_10H', 'Distancia_SMA20'
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

    # Entrenamos el modelo final con los parámetros ganadores de Optuna
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    print("📊 4. Validación Científica (Out-of-Sample)...")
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]
    
    acc = accuracy_score(y_test_final, y_pred) * 100
    auc = roc_auc_score(y_test_final, y_pred_proba) * 100
    report = classification_report(y_test_final, y_pred, target_names=['BAJA (0)', 'SUBE (1)'])
    
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

    if acc > 51.0 and auc > 51.0 and profit_factor >= 1.02:
        print("\n💾 5. Modelo aprobado por Criterio Múltiple. Actualizando entornos de producción...")
        joblib.dump(best_model, model_path)
        joblib.dump(features, features_path)
        print(f"   ✅ Archivo serializado con éxito: {model_path.name}")
    else:
        print("\n   ⚠️ ALERTA: El nuevo modelo no supera los criterios mínimos de calidad.")
        
        if model_path.exists() and features_path.exists():
            print(f"   🛡️ POLÍTICA FALLBACK ACTIVADA: Se mantiene el modelo anterior de {asset} operativo.")
            print("      Evitando paradas de producción. El bot mantendrá su configuración previa.")
        else:
            print(f"   ❌ AVISO: No existe un modelo previo en la carpeta. {asset} no operará esta semana.")

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
            print(f"❌ ERROR CRÍTICO: Fallo de login en MT5. Revisa tus credenciales en el .env.")
        else:
            for asset in ASSETS_TO_TRAIN:
                train_xgboost_for_asset(asset)
            
        mt5.shutdown()
        print("\n✅ ENTRENAMIENTO XGBOOST FINALIZADO.")