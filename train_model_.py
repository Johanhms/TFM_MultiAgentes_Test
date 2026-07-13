import os
import warnings
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import pandas_ta as ta
import xgboost as xgb
import optuna
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.mixture import GaussianMixture
 
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
# ============================== CONFIGURACIÓN ==============================
STRATEGY_TYPE = "LONG_SHORT"  # "LONG_SHORT" o "LONG_ONLY"
 
ASSETS_TO_TRAIN = ["BTC-USD", "EURUSD=X", "GBPUSD=X", "GC=F", "^GSPC", "CL=F", "^DJI", "NVDA"]
ASSET_MAPPING = {
    "BTC-USD": "BTCUSD", "EURUSD=X": "EURUSD.sml", "GBPUSD=X": "GBPUSD.sml",
    "GC=F": "XAUUSD.sml", "^GSPC": "US500", "CL=F": "USOIL.sml",
    "^DJI": "US30", "NVDA": "NVDA_CFD.US"
}
 
# Costo de transacción REALISTA por activo (bps, ida+vuelta aproximado por spread típico).
# AJUSTAR con datos reales de tu broker (spread promedio observado en MT5) antes de confiar en esto.
TRANSACTION_COST_BPS = {
    "BTC-USD": 6.0, "EURUSD=X": 1.2, "GBPUSD=X": 1.8, "GC=F": 3.5,
    "^GSPC": 2.0, "CL=F": 5.0, "^DJI": 2.5, "NVDA": 3.0,
}
DEFAULT_COST_BPS = 3.0
 
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
 
# Walk-forward purgado
N_FOLDS_WF = 4
EMBARGO_BARS = 15
MIN_TRAIN_FRAC = 0.55
MIN_TRADES_VALID = 6          # mínimo de trades para considerar válido un fold/threshold
N_OPTUNA_TRIALS = 25
MAX_HORIZON_SEARCH = 15
 
# Grid de thresholds a calibrar (probabilidad mínima de la clase BUY/SELL)
THRESH_GRID = np.round(np.arange(0.36, 0.62, 0.02), 2)
 
# Barrera triple: horizonte máximo de velas hacia adelante
MAX_HORIZON = 8
VOL_MULT = 0.6
 
# ---- 1. RE-ENTRENAMIENTO PERIÓDICO (walk-forward rolling en vivo) ----
# El script siempre entrena con las últimas n_velas disponibles (ventana rolling),
# pero esto evita re-entrenar cada vez que se corre el script sin necesidad
# (sobreajuste a ruido de corto plazo) y evita dejar un modelo "viejo" operando
# sin actualizarse. Se controla por la antigüedad del archivo guardado.
MODEL_MAX_AGE_DAYS = 14
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "0") == "1"

# ---- 2. MONITOREO DE DRIFT DE PROBABILIDADES PREDICHAS ----
# Se guarda un histograma de referencia (baseline) de prob_buy/prob_sell sobre el
# holdout. En vivo, trading_agents puede comparar la distribución reciente de
# probabilidades contra este baseline vía PSI (Population Stability Index) para
# detectar si el modelo está prediciendo fuera de su rango de calibración original.
PSI_N_BINS = 10

# ---- 3. VALIDACIÓN DE CORRELACIÓN ENTRE ACTIVOS (riesgo de portafolio concentrado) ----
# Matriz de correlación de retornos entre TODOS los activos del universo, para que
# el Risk Manager en vivo pueda limitar exposición agregada cuando varias posiciones
# abiertas están altamente correlacionadas (ej. EURUSD/GBPUSD, o US500/US30/NVDA
# en escenarios risk-on/risk-off).
CORRELATION_LOOKBACK_BARS = 500 
 
# ============================== FEATURES ==============================
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_20'] = df['Log_Ret'].rolling(window=20).std()
    df['Dist_EMA20'] = df['Close'] / df.ta.ema(length=20) - 1
    df['Dist_EMA50'] = df['Close'] / df.ta.ema(length=50) - 1
    df['RSI_14'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df['Volume_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
 
    adx_df = df.ta.adx(length=14)
    df['ADX_14'] = adx_df['ADX_14']
    df['ATR_14'] = df.ta.atr(length=14)
 
    macd_cols = macd.columns.tolist()
    features = ['Log_Ret', 'Vol_20', 'Dist_EMA20', 'Dist_EMA50', 'RSI_14',
                'Volume_ZScore', 'ADX_14'] + macd_cols
    return df, features
  
def attach_htf_context(df: pd.DataFrame, symbol_mt5: str, base_timeframe) -> tuple[pd.DataFrame, list[str]]:
    """Añade contexto de tendencia del timeframe superior (D1 si base=H1, W1 si base=D1)."""
    htf_tf = mt5.TIMEFRAME_D1 if base_timeframe == mt5.TIMEFRAME_H1 else mt5.TIMEFRAME_W1
    n_htf = 800
    velas_htf = mt5.copy_rates_from_pos(symbol_mt5, htf_tf, 0, n_htf)
    if velas_htf is None or len(velas_htf) < 60:
        df['HTF_Trend'] = 0.0
        df['HTF_EMA_Slope'] = 0.0
        return df, ['HTF_Trend', 'HTF_EMA_Slope']
 
    htf = pd.DataFrame(velas_htf)
    htf['time'] = pd.to_datetime(htf['time'], unit='s')
    htf = htf.rename(columns={'close': 'Close'})[['time', 'Close']].sort_values('time')
    htf['EMA50_HTF'] = htf['Close'].ewm(span=50, adjust=False).mean()
    htf['HTF_Trend'] = np.where(htf['Close'] > htf['EMA50_HTF'], 1.0, -1.0)
    htf['HTF_EMA_Slope'] = htf['EMA50_HTF'].pct_change(5)
 
    df_reset = df.reset_index().sort_values('time')
    merged = pd.merge_asof(df_reset, htf[['time', 'HTF_Trend', 'HTF_EMA_Slope']],
                            on='time', direction='backward')
    merged = merged.set_index('time')
    return merged, ['HTF_Trend', 'HTF_EMA_Slope']
 
# ============================== ETIQUETADO: TRIPLE BARRERA REAL ==============================
def label_triple_barrera(df: pd.DataFrame, cost_threshold: float,
                          vol_mult: float = VOL_MULT, max_horizon: int = MAX_HORIZON) -> pd.Series:
    """
    Barrera superior/inferior calculadas en t, y se recorre hacia adelante hasta
    max_horizon velas buscando el PRIMER toque (path-dependent). Si no se toca
    ninguna barrera dentro del horizonte -> HOLD (barrera vertical / timeout).
    Esto reemplaza la etiqueta de un solo paso del script original, que era
    indistinguible de ruido en H1.
    """
    n = len(df)
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    vol = df['Vol_20'].fillna(0).values
 
    barrier_width = np.maximum(vol * vol_mult, 0) + cost_threshold
    upper = close * (1 + barrier_width)
    lower = close * (1 - barrier_width)
 
    label = np.ones(n)  # default HOLD
    active = np.ones(n, dtype=bool)
    idx_all = np.arange(n)
 
    for h in range(1, max_horizon + 1):
        valid = active & (idx_all + h < n)
        idx = idx_all[valid]
        if len(idx) == 0:
            break
        fut_high = high[idx + h]
        fut_low = low[idx + h]
        hit_up = fut_high >= upper[idx]
        hit_down = fut_low <= lower[idx]
        both = hit_up & hit_down
        only_up = hit_up & ~both
        only_down = hit_down & ~both
 
        if both.any():
            close_h = close[idx[both] + h]
            mid = (upper[idx[both]] + lower[idx[both]]) / 2
            label[idx[both]] = np.where(close_h >= mid, 2, 0)
        label[idx[only_up]] = 2
        label[idx[only_down]] = 0
 
        resolved = only_up | only_down | both
        active[idx[resolved]] = False
 
    # Las últimas max_horizon filas no tienen ventana completa -> se descartan después.
    return pd.Series(label, index=df.index)
  
# ============================== WALK-FORWARD PURGADO ==============================
def purged_walk_forward_splits(n_samples: int, n_folds: int = N_FOLDS_WF,
                                embargo: int = EMBARGO_BARS, min_train_frac: float = MIN_TRAIN_FRAC):
    start_train_end = int(n_samples * min_train_frac)
    remaining = n_samples - start_train_end
    fold_size = remaining // (n_folds + 1)  # +1 reserva el holdout final
    splits = []
    for k in range(n_folds):
        val_start = start_train_end + k * fold_size
        val_end = val_start + fold_size
        if val_end > n_samples:
            break
        train_idx = np.arange(0, max(0, val_start - embargo))
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) < 100 or len(val_idx) < 20:
            continue
        splits.append((train_idx, val_idx))
    # Holdout final: nunca tocado por Optuna ni por calibración de thresholds.
    holdout_start = start_train_end + n_folds * fold_size + embargo
    holdout_idx = np.arange(min(holdout_start, n_samples), n_samples)
    return splits, holdout_idx
 
# ============================== EVALUACIÓN FINANCIERA ==============================
def evaluar_ventaja_financiera(y_pred_proba, df_slice, regimes_slice, active_regime,
                                asset, timeframe, cost_bps, thresh_buy, thresh_sell):
    prob_sell = y_pred_proba[:, 0]
    prob_buy = y_pred_proba[:, 2]
 
    señales = np.zeros(len(y_pred_proba))
    señales[prob_buy > thresh_buy] = 1
    if STRATEGY_TYPE == "LONG_SHORT":
        señales[prob_sell > thresh_sell] = -1
    señales = np.where(regimes_slice == active_regime, señales, 0)
 
    retornos_futuros = df_slice['Retorno_Forward'].values
    costo_unitario = (cost_bps / 10000) * 2  # entrada + salida de una posición
 
    señal_prev = np.concatenate(([0], señales[:-1]))
    cambio = np.abs(señales - señal_prev)  # 0, 1 (abrir/cerrar) o 2 (flip directo, doble costo)
    costo_friccion = cambio * costo_unitario
 
    retornos_barra = señales * retornos_futuros - costo_friccion
    retornos_barra = pd.Series(retornos_barra, index=df_slice.index).fillna(0.0)
 
    trades_ejecutados = int(np.sum(cambio > 0))
    retornos_activos = retornos_barra[señales != 0]
 
    if trades_ejecutados < 1:
        return 0.0, 0.0, 0.0, 0
 
    # Sharpe sobre la SERIE COMPLETA (incluye barras en cero), no solo barras activas.
    # Esto evita el error de anualizar como si cada barra activa fuera una observación anual.
    dias_por_anio = 365 if "BTC" in asset else 252
    factor_anualizacion = np.sqrt(dias_por_anio * 24) if timeframe == mt5.TIMEFRAME_H1 else np.sqrt(dias_por_anio)
    media = retornos_barra.mean()
    vol = retornos_barra.std()
    sharpe_ratio = (media / vol) * factor_anualizacion if vol > 0 else 0.0
 
    ganancias = retornos_activos[retornos_activos > 0].sum()
    perdidas = np.abs(retornos_activos[retornos_activos < 0].sum())
    profit_factor = ganancias / perdidas if perdidas > 0 else 0.0
 
    curva_equidad = (1 + retornos_barra).cumprod()
    picos = curva_equidad.cummax()
    drawdowns = (curva_equidad - picos) / picos
    max_drawdown = drawdowns.min() * 100 if len(drawdowns) > 0 else 0.0
 
    return sharpe_ratio, profit_factor, max_drawdown, trades_ejecutados

def find_best_thresholds(proba, df_slice, regimes_slice, active_regime, asset, timeframe, cost_bps):
    """Calibra thresholds de BUY/SELL maximizando Sharpe SOLO en el fold de validación dado."""
    best_score, best_tb, best_ts = -1e9, 0.42, 0.42
    ts_grid = THRESH_GRID if STRATEGY_TYPE == "LONG_SHORT" else [1.0]
    for tb in THRESH_GRID:
        for ts in ts_grid:
            sharpe, pf, mdd, trades = evaluar_ventaja_financiera(
                proba, df_slice, regimes_slice, active_regime, asset, timeframe, cost_bps, tb, ts
            )
            if trades < MIN_TRADES_VALID:
                continue
            score = sharpe  # se puede combinar con pf: score = sharpe * min(pf, 2.0)
            if score > best_score:
                best_score, best_tb, best_ts = score, tb, ts
    return best_score, best_tb, best_ts

# ============================== 1. GATE DE RE-ENTRENAMIENTO ==============================
def should_retrain(regime_path: Path, max_age_days: int = MODEL_MAX_AGE_DAYS) -> bool:
    """Evita re-entrenar si el modelo guardado sigue vigente y no se forzó vía FORCE_RETRAIN=1."""
    if FORCE_RETRAIN or not regime_path.exists():
        return True
    metadata = joblib.load(regime_path)
    trained_at = metadata.get('trained_at')
    if trained_at is None:
        return True
    age_days = (pd.Timestamp.utcnow() - pd.Timestamp(trained_at)).days
    return age_days >= max_age_days

# ============================== 2. DRIFT DE PROBABILIDADES (PSI) ==============================
def build_probability_baseline(proba_holdout, regimes_holdout, active_regime, n_bins: int = PSI_N_BINS):
    """
    Construye el histograma de referencia (baseline) de prob_buy y prob_sell sobre
    el holdout, filtrado al régimen activo (el mismo filtro que se usa en vivo).
    Este baseline se guarda junto al modelo y sirve para que, en producción,
    trading_agents compare la distribución reciente de probabilidades contra esta
    referencia y detecte drift (el modelo empieza a predecir fuera de su rango
    de calibración original, típicamente señal de que el régimen de mercado cambió
    y el modelo necesita re-entrenarse antes de lo programado).
    """
    mask = regimes_holdout == active_regime
    prob_buy = proba_holdout[mask, 2]
    prob_sell = proba_holdout[mask, 0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _hist(arr):
        counts, _ = np.histogram(arr, bins=bin_edges)
        # Normalizado a proporciones (no conteos crudos) para poder comparar con
        # ventanas de tamaño distinto en vivo.
        total = counts.sum()
        return (counts / total) if total > 0 else np.ones(n_bins) / n_bins

    return {
        'bin_edges': bin_edges,
        'prob_buy_baseline': _hist(prob_buy),
        'prob_sell_baseline': _hist(prob_sell),
    }

def compute_psi(baseline_props: np.ndarray, current_props: np.ndarray, epsilon: float = 1e-4) -> float:
    """
    Population Stability Index entre dos distribuciones ya expresadas como
    proporciones por bin. PSI < 0.1: sin drift relevante. 0.1-0.25: drift moderado
    (vigilar). > 0.25: drift significativo (recomendable re-entrenar / pausar).
    Esta función queda aquí para reutilizarse igual en trading_agents_OP2.py.
    """
    baseline = np.clip(baseline_props, epsilon, None)
    current = np.clip(current_props, epsilon, None)
    return float(np.sum((current - baseline) * np.log(current / baseline)))

# ============================== 3. CORRELACIÓN ENTRE ACTIVOS ==============================
def compute_and_save_correlation_matrix(assets: list, asset_mapping: dict, timeframe, suffix: str):
    """
    Calcula la matriz de correlación de retornos logarítmicos entre TODOS los
    activos del universo de entrenamiento, en el mismo timeframe, y la guarda para
    que el Risk Manager en vivo pueda evitar concentración de riesgo en posiciones
    correlacionadas (ej. abrir BUY en EURUSD y BUY en GBPUSD simultáneamente es,
    en la práctica, casi la misma apuesta apalancada dos veces).
    """
    print(f"\n🔗 Calculando matriz de correlación entre activos [{suffix}]...")
    retornos = {}
    for asset in assets:
        symbol_mt5 = asset_mapping.get(asset, asset)
        velas = mt5.copy_rates_from_pos(symbol_mt5, timeframe, 0, CORRELATION_LOOKBACK_BARS)
        if velas is None or len(velas) < 50:
            continue
        s = pd.DataFrame(velas)
        s['time'] = pd.to_datetime(s['time'], unit='s')
        s.set_index('time', inplace=True)
        retornos[asset] = np.log(s['close'] / s['close'].shift(1))

    if len(retornos) < 2:
        print("   ⚠️ Datos insuficientes para calcular correlación entre activos. Se omite.")
        return

    df_ret = pd.DataFrame(retornos).dropna(how='all')
    corr_matrix = df_ret.corr()

    corr_path = BASE_DIR / f'correlation_matrix_{suffix}.joblib'
    joblib.dump({
        'correlation_matrix': corr_matrix,
        'assets': list(corr_matrix.columns),
        'computed_at': pd.Timestamp.utcnow(),
    }, corr_path)
    print(f"   💾 Matriz de correlación guardada en {corr_path.name}")
 
# ============================== ENTRENAMIENTO ==============================
def train_xgboost_for_asset(asset: str, timeframe):
    symbol_mt5 = ASSET_MAPPING.get(asset, asset)
    tf_label = "D1_MACRO" if timeframe == mt5.TIMEFRAME_D1 else "H1_MICRO"
    suffix = "1d" if timeframe == mt5.TIMEFRAME_D1 else "1h"
    n_velas = 2000 if timeframe == mt5.TIMEFRAME_D1 else 15000
    cost_bps = TRANSACTION_COST_BPS.get(asset, DEFAULT_COST_BPS)
    cost_threshold = cost_bps / 10000

    safe_name = asset.replace("=X", "").replace("=F", "").replace("^", "")
    model_path = BASE_DIR / f'quant_model_{suffix}_{safe_name}.joblib'
    features_path = BASE_DIR / f'model_features_{suffix}_{safe_name}.joblib'
    regime_path = BASE_DIR / f'gmm_regime_{suffix}_{safe_name}.joblib'

    if not should_retrain(regime_path):
        print(f"\n⏭️  [{tf_label}] {asset}: modelo vigente (< {MODEL_MAX_AGE_DAYS} días). "
              f"Se omite re-entrenamiento (usa FORCE_RETRAIN=1 para forzar).")
        return

    print(f"\n{'='*85}")
    print(f"🚀 PIPELINE OP3 [{tf_label}] -> {asset} ({STRATEGY_TYPE}) | Costo: {cost_bps} bps")
    print(f"{'='*85}")

    velas_mt5 = mt5.copy_rates_from_pos(symbol_mt5, timeframe, 0, n_velas)
    if velas_mt5 is None or len(velas_mt5) < 800:
        print(f"❌ ERROR: Datos insuficientes en MT5 para {symbol_mt5}. Saltando...")
        return

    df = pd.DataFrame(velas_mt5)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'tick_volume']].rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}
    )

    df, base_features = build_features(df)
    df, htf_features = attach_htf_context(df, symbol_mt5, timeframe)
    features = base_features + htf_features

    df['Retorno_Forward'] = df['Close'].shift(-1) / df['Close'] - 1
    
    # 1. Limpiar NaNs iniciales
    df.dropna(inplace=True)
    
    # 2. Recortar Horizonte Máximo Absoluto para proteger la integridad de las etiquetas
    df = df.iloc[:-MAX_HORIZON_SEARCH] if len(df) > MAX_HORIZON_SEARCH else df

    n = len(df)
    splits, holdout_idx = purged_walk_forward_splits(n)
    if len(splits) < 2 or len(holdout_idx) < 50:
        print("❌ ERROR: Datos insuficientes para walk-forward purgado. Saltando...")
        return

    print("🧠 1. Optimizando Hiperparámetros y Barreras con Optuna (Cero Fuga de Datos)...")

    def objective(trial):
        # Barreras Dinámicas
        vol_mult = trial.suggest_float('vol_mult', 0.3, 1.5)
        max_horizon = trial.suggest_int('max_horizon', 5, MAX_HORIZON_SEARCH)
        
        y_dynamic = label_triple_barrera(df, cost_threshold, vol_mult, max_horizon)
        
        params = {
            'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
            'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'subsample': trial.suggest_float('subsample', 0.6, 0.85),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
        }

        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = df[features].iloc[train_idx], y_dynamic.iloc[train_idx]
            X_va, y_va = df[features].iloc[val_idx], y_dynamic.iloc[val_idx]
            
            # Penaliza configuraciones de barrera que colapsan clases
            if len(np.unique(y_tr)) < 3:
                raise optuna.exceptions.TrialPruned("Clases incompletas en entrenamiento debido a barreras extremas.")

            # GMM ajustado estrictamente dentro del fold (Sin mirar al futuro)
            gmm_cv = GaussianMixture(n_components=2, random_state=42, n_init=3)
            gmm_cv.fit(X_tr[['Vol_20', 'ADX_14']])
            
            df_reg = pd.DataFrame({'Regime': gmm_cv.predict(X_tr[['Vol_20', 'ADX_14']]), 'ADX': X_tr['ADX_14'].values})
            active_regime = df_reg.groupby('Regime')['ADX'].mean().idxmax()
            
            regimes_val = gmm_cv.predict(X_va[['Vol_20', 'ADX_14']])

            sw = compute_sample_weight(class_weight='balanced', y=y_tr)
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
            
            proba_va = model.predict_proba(X_va)
            score, _, _ = find_best_thresholds(
                proba_va, df.iloc[val_idx], regimes_val, active_regime, asset, timeframe, cost_bps
            )
            fold_scores.append(score)

        fold_scores = np.array(fold_scores)
        return fold_scores.mean() - 0.5 * fold_scores.std()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best_params = study.best_params
    best_vol_mult = best_params.pop('vol_mult')
    best_max_horizon = best_params.pop('max_horizon')
    
    best_params.update({'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
                         'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 20})

    print("🎯 2. Generando Target Final y Entrenando GMM Pre-Holdout...")
    
    df['Target'] = label_triple_barrera(df, cost_threshold, best_vol_mult, best_max_horizon)
    X = df[features]
    y = df['Target']
    
    train_final_idx = np.arange(0, holdout_idx[0])
    
    # Configuración final de GMM
    gmm_features = ['Vol_20', 'ADX_14']
    gmm_final = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gmm_final.fit(X.iloc[train_final_idx][gmm_features])
    
    regimes_full = gmm_final.predict(X[gmm_features])
    df_reg_final = pd.DataFrame({'Regime': regimes_full[train_final_idx], 'ADX': X.iloc[train_final_idx]['ADX_14'].values})
    active_regime_final = df_reg_final.groupby('Regime')['ADX'].mean().idxmax()

    print("🎯 3. Calibrando thresholds finales en el último fold de validación...")
    last_train_idx, last_val_idx = splits[-1]
    sw_last = compute_sample_weight(class_weight='balanced', y=y.iloc[last_train_idx])
    
    calib_model = xgb.XGBClassifier(**{k: v for k, v in best_params.items() if k != 'early_stopping_rounds'})
    calib_model.fit(X.iloc[last_train_idx], y.iloc[last_train_idx], sample_weight=sw_last, verbose=False)
    proba_calib = calib_model.predict_proba(X.iloc[last_val_idx])
    
    _, thresh_buy, thresh_sell = find_best_thresholds(
        proba_calib, df.iloc[last_val_idx], regimes_full[last_val_idx], active_regime_final, asset, timeframe, cost_bps
    )
    print(f"      Thresholds Calibrados -> BUY: {thresh_buy} | SELL: {thresh_sell}")

    print("🏋️ 4. Entrenamiento Final con Early Stopping (Datos Pre-Holdout)...")
    sw_final = compute_sample_weight(class_weight='balanced', y=y.iloc[train_final_idx])
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(
        X.iloc[train_final_idx], y.iloc[train_final_idx], sample_weight=sw_final,
        eval_set=[(X.iloc[last_val_idx], y.iloc[last_val_idx])], verbose=False
    )

    print("📊 5. Validación en HOLDOUT puro...")
    proba_holdout = best_model.predict_proba(X.iloc[holdout_idx])
    sharpe, profit_factor, max_dd, trades = evaluar_ventaja_financiera(
        proba_holdout, df.iloc[holdout_idx], regimes_full[holdout_idx], active_regime_final,
        asset, timeframe, cost_bps, thresh_buy, thresh_sell
    )

    print(f"\n   💰 RENDIMIENTO EN HOLDOUT:")
    print(f"      Net Sharpe Ratio:              {sharpe:.2f}")
    print(f"      Net Profit Factor:             {profit_factor:.2f}")
    print(f"      Max Drawdown:                  {max_dd:.2f}%")
    print(f"      Trades Ejecutados:             {trades}")
    print(f"      Mult Volatilidad Óptimo:       {best_vol_mult:.2f}")
    print(f"      Horizonte Máx Óptimo:          {best_max_horizon}")

    if sharpe > 0.3 and profit_factor > 1.05 and trades >= 8:
        print(f"\n💾 6. El modelo pasa la validación en holdout puro. Guardando para producción...")

        # Baseline de probabilidades para monitoreo de drift en vivo (recomendación #2).
        prob_baseline = build_probability_baseline(proba_holdout, regimes_full[holdout_idx], active_regime_final)

        joblib.dump(best_model, model_path)
        joblib.dump(features, features_path)
        joblib.dump({
            'gmm': gmm_final, 'active_regime': active_regime_final, 'gmm_features': gmm_features,
            'threshold_buy': float(thresh_buy), 'threshold_sell': float(thresh_sell),
            'cost_bps': cost_bps, 'strategy_type': STRATEGY_TYPE,
            'optimal_vol_mult': best_vol_mult, 'optimal_max_horizon': best_max_horizon,
            # --- Recomendación #1: metadata para el gate de re-entrenamiento periódico ---
            'trained_at': pd.Timestamp.utcnow(),
            # --- Recomendación #2: baseline para monitoreo de drift (PSI) en vivo ---
            'prob_baseline': prob_baseline,
        }, regime_path)
    else:
        print(f"\n   ⚠️ ALERTA: No supera el gate en holdout puro. Modelo descartado.")

if __name__ == "__main__":
    load_dotenv()
    if mt5.initialize(path=os.getenv("MT5_PATH")):
        if mt5.login(login=int(os.getenv("MT5_LOGIN")), password=os.getenv("MT5_PASSWORD"), server=os.getenv("MT5_SERVER")):
            for asset in ASSETS_TO_TRAIN:
                train_xgboost_for_asset(asset, mt5.TIMEFRAME_D1)
                train_xgboost_for_asset(asset, mt5.TIMEFRAME_H1)

            # Recomendación #3: matriz de correlación entre activos, una vez por timeframe,
            # para que el Risk Manager en vivo pueda limitar exposición concentrada.
            compute_and_save_correlation_matrix(ASSETS_TO_TRAIN, ASSET_MAPPING, mt5.TIMEFRAME_D1, "1d")
            compute_and_save_correlation_matrix(ASSETS_TO_TRAIN, ASSET_MAPPING, mt5.TIMEFRAME_H1, "1h")
        mt5.shutdown()