"""
================================================================================
BITCOIN MACRO-QUANT PREDICTOR
================================================================================


import optuna
import xgboost as xgb
from sklearn.metrics import (log_loss, accuracy_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ANSI Colors for Terminal UI
GRN = "\033[92m"
RED = "\033[91m"
BLU = "\033[94m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"
RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"

# ──────────────────────────────────────────────
# 1. DATA ACQUISITION (EXTENDED)
# ──────────────────────────────────────────────


def fetch_macro_data(days: int = 4000) -> pd.DataFrame:
    print(
        f"\n{CYN}→ Fetching Market Data (BTC, ETH, SPY, DXY, VIX, Gold, TLT, BITO)...{RST}")

    try:
        btc = yf.Ticker(
            "BTC-USD").history(period="max")[["Open", "High", "Low", "Close", "Volume"]]
        if btc.empty:
            raise ValueError("Empty BTC data")
    except Exception as e:
        raise ValueError(f"Failed BTC-USD: {e}")

    tickers = {
        "SPY_Close": "SPY",
        "DXY_Close": "DX-Y.NYB",
        "VIX_Close": "^VIX",
        "Gold_Close": "GC=F",
        "TLT_Close": "TLT",
        "ETH_Close": "ETH-USD",
        "BITO_Volume": None,   # handled separately
    }

    macro_dfs = []
    for col_name, ticker in tickers.items():
        if ticker is None:
            continue
        try:
            data = yf.Ticker(ticker).history(period="max")
            if col_name == "BITO_Volume":
                pass  # skip
            else:
                t = data[["Close"]].rename(columns={"Close": col_name})
                t.index = t.index.tz_localize(None).normalize()
                macro_dfs.append(t)
        except Exception as e:
            print(f"{YEL}  Warning: {ticker}: {e}{RST}")

    # BITO (Bitcoin ETF) — institutional flow proxy
    try:
        bito = yf.Ticker("BITO").history(period="max")[
            ["Volume"]].rename(columns={"Volume": "BITO_Vol"})
        bito.index = bito.index.tz_localize(None).normalize()
        macro_dfs.append(bito)
    except:
        print(f"{YEL}  Warning: BITO not available{RST}")

    btc.index = btc.index.tz_localize(None).normalize()
    df = btc
    for mdf in macro_dfs:
        df = df.join(mdf, how="left")
    df = df.ffill().dropna()
    return df[df["Volume"] > 0].tail(days)

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING (30+ FEATURES)
# ──────────────────────────────────────────────


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd_hist(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def compute_bollinger(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return (upper - lower) / (sma + 1e-10), (series - lower) / (upper - lower + 1e-10)


def compute_stochastic(high, low, close, period=14):
    lo = low.rolling(period).min()
    hi = high.rolling(period).max()
    return 100 * (close - lo) / (hi - lo + 1e-10)


def add_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def compute_synthetic_fear_greed(df):
    components = []

    # 1. Volatility component (high vol = fear)
    vol_20 = df['LogRet_1d'].rolling(20).std() * np.sqrt(365)
    vol_z = (vol_20 - vol_20.rolling(90).mean()) / \
        (vol_20.rolling(90).std() + 1e-10)
    vol_score = 50 - (vol_z.clip(-2, 2) * 25)
    components.append(vol_score)

    # 2. Momentum component (RSI-based)
    rsi = compute_rsi(df['Close'], 14)
    mom_score = rsi
    components.append(mom_score)

    # 3. Volume surge component
    vol_ratio = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)
    direction = np.sign(df['Close'].pct_change(1))
    vol_sent = vol_ratio * direction
    vol_sent_z = (vol_sent - vol_sent.rolling(90).mean()) / \
        (vol_sent.rolling(90).std() + 1e-10)
    vol_sent_score = 50 + (vol_sent_z.clip(-2, 2) * 25)
    components.append(vol_sent_score)

    # 4. Price vs SMA (trend following = greed)
    sma_50 = df['Close'].rolling(50).mean()
    dist = (df['Close'] - sma_50) / (sma_50 + 1e-10)
    dist_score = 50 + (dist.clip(-0.3, 0.3) / 0.3 * 50)
    components.append(dist_score)

    fg = pd.concat(components, axis=1).mean(axis=1)
    return fg.clip(0, 100)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Ret_1d"] = df["Close"].pct_change(1)
    df["Ret_3d"] = df["Close"].pct_change(3)
    df["Ret_5d"] = df["Close"].pct_change(5)
    df["Ret_10d"] = df["Close"].pct_change(10)
    df["Ret_20d"] = df["Close"].pct_change(20)
    df["LogRet_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close],
                          axis=1).max(axis=1).rolling(14).mean()
    df['ATR_Norm'] = df['ATR'] / df['Close']
    df['RVol_5d'] = df['LogRet_1d'].rolling(5).std() * np.sqrt(365)
    df['RVol_20d'] = df['LogRet_1d'].rolling(20).std() * np.sqrt(365)
    df['Vol_Ratio'] = df['RVol_5d'] / (df['RVol_20d'] + 1e-10)

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Price_vs_SMA20"] = (df["Close"] - df["SMA_20"]
                            ) / (df["SMA_20"] + 1e-10)
    df["Price_vs_SMA50"] = (df["Close"] - df["SMA_50"]
                            ) / (df["SMA_50"] + 1e-10)
    df["SMA20_vs_SMA50"] = (df["SMA_20"] - df["SMA_50"]
                            ) / (df["SMA_50"] + 1e-10)

    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_7'] = compute_rsi(df['Close'], 7)
    df['MACD_Hist'] = compute_macd_hist(df['Close']) / (df['Close'] + 1e-10)
    df['Stoch_K'] = compute_stochastic(df['High'], df['Low'], df['Close'])
    df['BB_Width'], df['BB_Pct'] = compute_bollinger(df['Close'])

    df['Vol_Ratio_20'] = df['Volume'] / \
        (df['Volume'].rolling(20).mean() + 1e-10)
    obv_dir = np.sign(df['Close'].diff())
    df['OBV_ROC'] = (obv_dir * df['Volume']).cumsum().pct_change(5)

    df['ADX_14'] = add_adx(df['High'], df['Low'], df['Close'], 14)

    df['Fear_Greed'] = compute_synthetic_fear_greed(df)
    df['FG_Change_5d'] = df['Fear_Greed'].diff(5)

    if 'ETH_Close' in df.columns:
        df['ETH_BTC_Ratio'] = df['ETH_Close'] / df['Close']
        df['ETH_BTC_Change'] = df['ETH_BTC_Ratio'].pct_change(5)

    if 'BITO_Vol' in df.columns:
        df['BITO_Vol_Ratio'] = df['BITO_Vol'] / \
            (df['BITO_Vol'].rolling(20).mean() + 1e-10)

    if 'SPY_Close' in df.columns:
        df["SPY_Ret_5d"] = df["SPY_Close"].pct_change(5)
        df["Corr_BTC_SPY"] = df["Ret_1d"].rolling(
            30).corr(df["SPY_Close"].pct_change(1))
    if 'DXY_Close' in df.columns:
        df["DXY_Ret_5d"] = df["DXY_Close"].pct_change(5)
    if 'VIX_Close' in df.columns:
        df["VIX_Level"] = df["VIX_Close"]
        df["VIX_Chg_5d"] = df["VIX_Close"].pct_change(5)
    if 'Gold_Close' in df.columns:
        df["Gold_Ret_5d"] = df["Gold_Close"].pct_change(5)
    if 'TLT_Close' in df.columns:
        df["TLT_Ret_5d"] = df["TLT_Close"].pct_change(5)

    for col in ['Ret_5d', 'RSI_14', 'Vol_Ratio', 'BB_Pct']:
        if col in df.columns:
            roll_mean = df[col].rolling(60).mean()
            roll_std = df[col].rolling(60).std()
            df[f'{col}_Z'] = (df[col] - roll_mean) / (roll_std + 1e-10)

    conditions = [
        (df['ADX_14'] < 20),
        (df['ADX_14'] >= 20) & (df['Close'] > df['SMA_50']),
        (df['ADX_14'] >= 20) & (df['Close'] <= df['SMA_50'])
    ]
    df['Regime'] = np.select(
        conditions, ['Range', 'Bull', 'Bear'], default='Range')

    return df


def get_feature_cols(df):
    candidates = [
        'Ret_1d', 'Ret_3d', 'Ret_5d', 'Ret_10d', 'Ret_20d', 'LogRet_1d',
        'ATR_Norm', 'RVol_5d', 'RVol_20d', 'Vol_Ratio',
        'Price_vs_SMA20', 'Price_vs_SMA50', 'SMA20_vs_SMA50',
        'RSI_14', 'RSI_7', 'MACD_Hist', 'Stoch_K', 'BB_Width', 'BB_Pct',
        'Vol_Ratio_20', 'OBV_ROC', 'ADX_14',
        'Fear_Greed', 'FG_Change_5d',
        'ETH_BTC_Ratio', 'ETH_BTC_Change',
        'BITO_Vol_Ratio',
        'SPY_Ret_5d', 'Corr_BTC_SPY', 'DXY_Ret_5d',
        'VIX_Level', 'VIX_Chg_5d', 'Gold_Ret_5d', 'TLT_Ret_5d',
        'Ret_5d_Z', 'RSI_14_Z', 'Vol_Ratio_Z', 'BB_Pct_Z',
    ]
    return [f for f in candidates if f in df.columns]

# ──────────────────────────────────────────────
# 3. TRIPLE BARRIER (YOUR TC — UNCHANGED)
# ──────────────────────────────────────────────


def build_triple_barrier(df, horizon=10, sl_mult=1.5, tp_mult=2.25):
    df = df.copy()
    if horizon > 0:
        print(
            f"{YEL}→ Triple Barrier (H={horizon}, SL={sl_mult}x, TP={tp_mult}x)...{RST}")

    targets = np.zeros(len(df))
    long_pnls, short_pnls = np.zeros(len(df)), np.zeros(len(df))
    long_res = ["⏱️ Time Exit"] * len(df)
    short_res = ["⏱️ Time Exit"] * len(df)

    for i in range(len(df)):
        if i + horizon >= len(df):
            continue
        entry = df['Close'].iloc[i]
        atr_pct = max(df['ATR_Norm'].iloc[i], 0.01)
        tp_dist = atr_pct * tp_mult
        sl_dist = atr_pct * sl_mult

        l_pnl = (df['Close'].iloc[i + horizon] / entry) - 1
        s_pnl = 1 - (df['Close'].iloc[i + horizon] / entry)
        l_out, s_out = "⏱️ Time Exit", "⏱️ Time Exit"
        label = 0

        for j in range(1, horizon + 1):
            h, l = df['High'].iloc[i + j], df['Low'].iloc[i + j]
            if l_out == "⏱️ Time Exit":
                if l <= entry * (1 - sl_dist):
                    l_pnl, l_out = -sl_dist, "🛑 Hit SL"
                elif h >= entry * (1 + tp_dist):
                    l_pnl, l_out = tp_dist, "🎯 Hit TP"
                    if label == 0:
                        label = 1
            if s_out == "⏱️ Time Exit":
                if h >= entry * (1 + sl_dist):
                    s_pnl, s_out = -sl_dist, "🛑 Hit SL"
                elif l <= entry * (1 - tp_dist):
                    s_pnl, s_out = tp_dist, "🎯 Hit TP"
                    if label == 0:
                        label = 2

        targets[i] = label
        long_pnls[i], short_pnls[i] = l_pnl, s_pnl
        long_res[i], short_res[i] = l_out, s_out

    df['Target_Class'] = targets
    df['Entry_Price'] = df['Close']
    df['Long_PnL'], df['Long_Outcome'] = long_pnls, long_res
    df['Short_PnL'], df['Short_Outcome'] = short_pnls, short_res
    df['Year'] = df.index.year
    df['Label_Long'] = (df['Target_Class'] == 1).astype(int)
    df['Label_Short'] = (df['Target_Class'] == 2).astype(int)

    if horizon == 0:
        return df
    return df.iloc[:-horizon]

# ──────────────────────────────────────────────
# 4. BINARY MODEL TRAINER
# ──────────────────────────────────────────────


def train_binary_model(X_train, y_train, X_test, y_test, label_name, n_trials=30, seed=42):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    w_pos = n_neg / max(n_pos, 1)
    sample_w = np.where(y_train == 1, w_pos, 1.0)

    print(
        f"  {label_name}: {int(n_pos)} pos ({n_pos/len(y_train)*100:.1f}%) | w={w_pos:.2f}")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
            'gamma': trial.suggest_float('gamma', 1.0, 8.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'n_estimators': 300,
            'early_stopping_rounds': 25,
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            'random_state': seed,
            'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_w,
                  eval_set=[(X_test, y_test)], verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            return roc_auc_score(y_test, y_proba)
        except:
            return 0.5

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        'n_estimators': 500,
        'early_stopping_rounds': 25,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'random_state': seed,
        'verbosity': 0,
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, sample_weight=sample_w,
              eval_set=[(X_test, y_test)], verbose=False)
    y_proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5

    print(f"    AUC={auc:.4f}")
    return model, y_proba, auc

# ──────────────────────────────────────────────
# RISK & DRAWDOWN FUNCTIONS (TERMINAL UI SUPPORT)
# ──────────────────────────────────────────────


def sharpe_with_ci(rets, ann=365):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 2 or rets.std() < 1e-10:
        return 0.0, -np.inf, np.inf
    sr_d = rets.mean() / rets.std()
    sr_a = sr_d * np.sqrt(ann)
    se_a = np.sqrt((1 + 0.5*sr_d**2) / max(len(rets)-1, 1)) * np.sqrt(ann)
    return sr_a, sr_a - 1.96*se_a, sr_a + 1.96*se_a


def drawdown_stats(daily_rets):
    rets = np.asarray(daily_rets, dtype=float)
    wealth = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    max_dur, cur = 0, 0
    for v in (dd < 0):
        cur = cur+1 if v else 0
        max_dur = max(max_dur, cur)
    trough = int(np.argmin(dd)) if len(dd) > 0 else 0
    recovered = np.where(wealth[trough:] >= peak[trough])[0]
    recovery = int(recovered[0]) if len(recovered) else len(rets) - trough
    ann_ret = float(
        (wealth[-1] ** (365 / max(len(rets), 1))) - 1) if len(rets) > 0 else 0
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
    return dict(max_dd=max_dd, avg_dd=avg_dd, max_dur=max_dur, recovery=recovery, calmar=calmar, wealth=wealth, dd_series=dd)

# ──────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────


def main():
    # ══════════════════════════════════════════
    # CONFIG
    # ══════════════════════════════════════════
    HORIZON = 10
    SL_MULT = 1.5
    TP_MULT = 2.25
    FEE = 0.004
    PURGE_DAYS = 15
    OPTUNA_TRIALS = 30
    LONG_THRESHOLD = 0.43
    SHORT_THRESHOLD = 0.43
    RANGE_THRESHOLD = 0.48
    RISK_PER_TRADE = 0.02
    COOLDOWN_AFTER = 3

    print("\n" + "═"*78)
    print(f"{BLD}  BITCOIN XGBOOST MACRO-QUANT  —  V17.4{RST}")
    print(
        f"  REGIME-ALIGNED + POSITION SIZING ({RISK_PER_TRADE:.0%} risk/trade)")
    print(f"  H={HORIZON}d | SL={SL_MULT}x | TP={TP_MULT}x | Fee={FEE*100:.1f}%")
    print(
        f"  Bull→Long@{LONG_THRESHOLD:.0%} | Bear→Short@{SHORT_THRESHOLD:.0%} | Range→Both@{RANGE_THRESHOLD:.0%}")
    print("═"*78)

    # ── DATA ──
    print(f"\n{BLD}[1/3] DATA & FEATURES{RST}")
    df_raw = fetch_macro_data(4000)
    print(f"{GRN}✓ {len(df_raw)} days{RST}")

    df = build_features(df_raw)
    FEATURES = get_feature_cols(df)
    print(f"{GRN}✓ {len(FEATURES)} features engineered{RST}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df = build_triple_barrier(
        df, horizon=HORIZON, sl_mult=SL_MULT, tp_mult=TP_MULT)

    n0 = (df['Target_Class'] == 0).sum()
    n1 = (df['Target_Class'] == 1).sum()
    n2 = (df['Target_Class'] == 2).sum()
    print(f"{YEL}  Target Breakdown: Noise={n0}  Long={n1}  Short={n2}{RST}")

    # ══════════════════════════════════════════
    # WALK-FORWARD CV
    # ══════════════════════════════════════════
    print(f"\n{BLD}[2/3] PURGED WALK-FORWARD CV{RST}")

    test_years = sorted([y for y in df['Year'].unique() if y >= 2022])
    test_data_all = []
    imp_l_all, imp_s_all = [], []
    final_long_model = final_short_model = final_scaler = None

    for test_year in test_years:
        print(f"\n{CYN}═ Fold: {test_year} ═{RST}")

        test_df = df[df['Year'] == test_year].copy()
        if len(test_df) < 10:
            continue

        test_start = test_df.index.min()
        purge_cutoff = test_start - pd.Timedelta(days=PURGE_DAYS)
        train_df = df[df.index < purge_cutoff].copy()
        if len(train_df) < 200:
            continue

        X_train = train_df[FEATURES].values
        X_test = test_df[FEATURES].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # LONG detector
        long_model, p_long, _ = train_binary_model(
            X_train_s, train_df['Label_Long'].values,
            X_test_s, test_df['Label_Long'].values,
            "LONG", n_trials=OPTUNA_TRIALS, seed=42 + test_year)

        # SHORT detector
        short_model, p_short, _ = train_binary_model(
            X_train_s, train_df['Label_Short'].values,
            X_test_s, test_df['Label_Short'].values,
            "SHORT", n_trials=OPTUNA_TRIALS, seed=99 + test_year)

        test_df['P_Long'] = p_long
        test_df['P_Short'] = p_short
        test_data_all.append(test_df.copy())

        final_long_model = long_model
        final_short_model = short_model
        final_scaler = scaler

        for feat, imp in zip(FEATURES, long_model.feature_importances_):
            imp_l_all.append({'feature': feat, 'importance': imp})
        for feat, imp in zip(FEATURES, short_model.feature_importances_):
            imp_s_all.append({'feature': feat, 'importance': imp})

    if not test_data_all:
        print(f"{RED}No data.{RST}")
        return

    # ══════════════════════════════════════════
    # REGIME-ALIGNED + RISK-MANAGED SIGNALS
    # ══════════════════════════════════════════
    print(f"\n{BLD}[3/3] REGIME-ALIGNED EXECUTION{RST}")

    trade_df = pd.concat(test_data_all).sort_index()
    trade_df['SL_Pct'] = trade_df['ATR_Norm'] * SL_MULT
    trade_df['TP_Pct'] = trade_df['ATR_Norm'] * TP_MULT

    signals, pnls, outcomes, pred_classes, pos_sizes = [], [], [], [], []
    consecutive_sl = 0

    for idx, row in trade_df.iterrows():
        p_l = row['P_Long']
        p_s = row['P_Short']
        regime = row['Regime']
        sl_pct = row['SL_Pct']

        sig = 0

        if consecutive_sl >= COOLDOWN_AFTER:
            sig = 0
            consecutive_sl = 0
        else:
            if regime == 'Bull':
                if p_l > LONG_THRESHOLD:
                    sig = 1
            elif regime == 'Bear':
                if p_s > SHORT_THRESHOLD:
                    sig = -1
            else:
                if p_l > RANGE_THRESHOLD and p_l > p_s:
                    sig = 1
                elif p_s > RANGE_THRESHOLD and p_s > p_l:
                    sig = -1

        position_size = min(RISK_PER_TRADE / (sl_pct + 1e-10), 1.0)

        if sig == 1:
            raw_pnl = row['Long_PnL'] - FEE
            pnls.append(raw_pnl * position_size)
            outcomes.append(row['Long_Outcome'])
            pred_classes.append(1)
            pos_sizes.append(position_size)
            if row['Long_Outcome'] == "🛑 Hit SL":
                consecutive_sl += 1
            else:
                consecutive_sl = 0
        elif sig == -1:
            raw_pnl = row['Short_PnL'] - FEE
            pnls.append(raw_pnl * position_size)
            outcomes.append(row['Short_Outcome'])
            pred_classes.append(2)
            pos_sizes.append(position_size)
            if row['Short_Outcome'] == "🛑 Hit SL":
                consecutive_sl += 1
            else:
                consecutive_sl = 0
        else:
            pnls.append(0.0)
            outcomes.append("➖ Cash")
            pred_classes.append(0)
            pos_sizes.append(0.0)

        signals.append(sig)

    trade_df['Signal'] = signals
    trade_df['Strategy_Return'] = pnls
    trade_df['Trade_Outcome'] = outcomes
    trade_df['Pred_Class'] = pred_classes
    trade_df['Pos_Size'] = pos_sizes

    # ══════════════════════════════════════════
    # OVERALL PREDICTIVE METRICS
    # ══════════════════════════════════════════
    try:
        auc_l = roc_auc_score(trade_df['Label_Long'], trade_df['P_Long'])
    except:
        auc_l = 0.5
    try:
        auc_s = roc_auc_score(trade_df['Label_Short'], trade_df['P_Short'])
    except:
        auc_s = 0.5

    y_true = trade_df['Target_Class'].values
    y_pred = trade_df['Pred_Class'].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ══════════════════════════════════════════
    # 180-DAY SURGICAL METRICS & DRAWDOWN
    # ══════════════════════════════════════════
    # Slice exactly the last 180 days for the ledger and performance stats
    df_180 = trade_df.tail(180).copy()

    exec_df_180 = df_180[df_180['Signal'] != 0]
    n_trades_180 = len(exec_df_180)
    wins_180 = exec_df_180[exec_df_180['Strategy_Return'] > 0]
    losses_180 = exec_df_180[exec_df_180['Strategy_Return'] < 0]

    win_rate_180 = len(wins_180) / max(n_trades_180, 1)
    avg_win_180 = wins_180['Strategy_Return'].mean() if len(
        wins_180) > 0 else 0
    avg_loss_180 = losses_180['Strategy_Return'].mean() if len(
        losses_180) > 0 else 0
    pf_180 = (avg_win_180 * len(wins_180)) / \
        (abs(avg_loss_180) * len(losses_180) + 1e-9)

    cum_strat_180 = (1 + df_180['Strategy_Return']).cumprod() - 1
    cum_bh_180 = (1 + df_180['Ret_1d']).cumprod() - 1

    sh_net_180, sh_net_lo_180, sh_net_hi_180 = sharpe_with_ci(
        df_180['Strategy_Return'])
    sh_bh_180, sh_bh_lo_180, sh_bh_hi_180 = sharpe_with_ci(df_180['Ret_1d'])
    dd_180 = drawdown_stats(df_180['Strategy_Return'])
    dd_bh_180 = drawdown_stats(df_180['Ret_1d'])
    total_fees_paid_180 = (FEE * exec_df_180['Pos_Size']).sum()

    # ══════════════════════════════════════════
    # TERMINAL UI REPORT
    # ══════════════════════════════════════════
    print("\n" + "═"*115)
    print(
        f"{BLD}  MODEL PERFORMANCE REPORT  —  XGBOOST MACRO-QUANT ({test_years[0]}–{test_years[-1]}){RST}")
    print("═"*115)

    print(f"\n  {'METRIC':<20} {'TEST SCORE':>11}")
    print(f"  {'─'*20} {'─'*11}")
    for lbl, val in [('Accuracy', acc), ('Precision (Macro)', prec), ('F1-Score (Macro)', f1),
                     ('ROC-AUC (Long)', auc_l), ('ROC-AUC (Short)', auc_s)]:
        v_pct = val * 100
        col_c = GRN if v_pct >= 55 else (YEL if v_pct >= 50 else RED)
        print(f"  {lbl:<20} {col_c}{v_pct:>10.1f}%{RST}")

    print(f"\n  Confusion Matrix (TEST):")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(f"  {'':14} Pred:Cash  Pred:Long  Pred:Short")
    print(f"  True:Noise    {cm[0, 0]:>6}    {cm[0, 1]:>7}    {cm[0, 2]:>8}")
    print(f"  True:Long     {cm[1, 0]:>6}    {cm[1, 1]:>7}    {cm[1, 2]:>8}")
    print(f"  True:Short    {cm[2, 0]:>6}    {cm[2, 1]:>7}    {cm[2, 2]:>8}")

    print("\n" + "═"*115)
    print(f"{BLD}  180-DAY BACKTEST LEDGER  |  Risk Sizing · Fee={FEE*100:.2f}% · TP={TP_MULT:.1f}x ATR · SL={SL_MULT:.1f}x ATR{RST}")
    print("═"*115)

    hdr = (f"{'DATE':<12} {'REGIME':<8} {'PRICE':>9} {'PROB%':>6} {'SIG':<7} {'SIZE':>6} {'SL$':>9} {'TP$':>9} {'NET%':>7} {'CUM%':>8}  ST")
    print(f"\n  {hdr}")
    print("  " + "─"*110)

    for i, (date, row) in enumerate(df_180.iterrows()):
        sig = row['Signal']
        pnl = row['Strategy_Return'] * 100
        cum = cum_strat_180.iloc[i] * 100
        daily_price = row['Entry_Price']
        sl_pct, tp_pct = row['SL_Pct'], row['TP_Pct']

        conf = row['P_Long'] if sig == 1 else (
            row['P_Short'] if sig == -1 else max(row['P_Long'], row['P_Short']))

        if sig == 1:
            lbl_sig, col = "LONG  ", GRN
            entry = daily_price
            tp_p, sl_p = entry*(1+tp_pct), entry*(1-sl_pct)
        elif sig == -1:
            lbl_sig, col = "SHORT ", RED
            entry = daily_price
            tp_p, sl_p = entry*(1-tp_pct), entry*(1+sl_pct)
        else:
            lbl_sig, col = "CASH  ", BLU
            entry = tp_p = sl_p = 0

        sl_s = f"{sl_p:>9,.0f}" if sl_p > 0 else "        —"
        tp_s = f"{tp_p:>9,.0f}" if tp_p > 0 else "        —"
        stat = f"{DIM}–{RST}" if sig == 0 else (
            f"{GRN}✓{RST}" if pnl > 0 else f"{RED}✗{RST}")
        nc = GRN if pnl > 0 else (RED if pnl < 0 else DIM)
        cc = GRN if cum > 0 else RED

        print(
            f"  {date.strftime('%Y-%m-%d'):<12} {row['Regime']:<8} ${daily_price:>8,.0f} {conf*100:>5.1f}% {col}{lbl_sig}{RST} {row['Pos_Size']:>5.3f}x {DIM}{sl_s}{RST} {DIM}{tp_s}{RST} {nc}{pnl:>6.2f}%{RST} {cc}{cum:>7.2f}%{RST}  {stat}")

    print("  " + "─"*110)

    def pline(label, val, fmt='%', col=None, extra=''):
        if col is None:
            col = GRN if val >= 0 else RED
        v_s = f"{val*100:>+8.2f}%" if fmt == '%' else f"{val:>+8.2f}"
        print(f"  {label:<38} {col}{v_s}{RST}{extra}")

    print()
    pline("Strategy Return (net of Fees)", cum_strat_180.iloc[-1])
    pline("Total Fees Impact", -total_fees_paid_180, col=RED)
    pline("Buy & Hold Return", cum_bh_180.iloc[-1])
    pline("Alpha (net vs B&H)", cum_strat_180.iloc[-1]-cum_bh_180.iloc[-1])

    print()
    print(
        f"  {'Sharpe Ratio (net, ann.)':<38} {(GRN if sh_net_180 > 0 else RED)}{sh_net_180:>+8.2f}{RST}  (95% CI: [{sh_net_lo_180:+.2f}, {sh_net_hi_180:+.2f}])")
    print(
        f"  {'Sharpe Ratio (Buy & Hold, ann.)':<38} {(GRN if sh_bh_180 > 0 else RED)}{sh_bh_180:>+8.2f}{RST}  (95% CI: [{sh_bh_lo_180:+.2f}, {sh_bh_hi_180:+.2f}])")
    print(
        f"  {'Max Drawdown (strategy)':<38} {RED}{dd_180['max_dd']*100:>+8.2f}%{RST}")
    print(
        f"  {'Calmar Ratio (strategy)':<38} {(GRN if dd_180['calmar'] > 0 else RED)}{dd_180['calmar']:>+8.2f}{RST}")

    print()
    win_c = GRN if win_rate_180 >= 0.5 else YEL
    print(f"  {'Win Rate (active signals)':<38} {win_c}{len(wins_180)}/{max(n_trades_180, 1)} = {win_rate_180*100:.1f}%{RST}")
    print(f"  {'Avg Winning Trade':<38} {GRN}{avg_win_180*100:>+8.3f}%{RST}")
    print(f"  {'Avg Losing Trade':<38} {RED}{avg_loss_180*100:>+8.3f}%{RST}")
    print(
        f"  {'Profit Factor':<38} {(GRN if pf_180 >= 1 else RED)}{pf_180:>8.2f}x{RST}")
    print(f"  {'Active Days / Total Days':<38}  {n_trades_180:>4}/180")

    # ── LIVE OUTLOOK / TOMORROW ──
    live_df = build_features(df_raw.tail(250))
    live_df = live_df.replace(
        [np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    lr = live_df.iloc[-1:]

    X_live = final_scaler.transform(lr[FEATURES].values)
    pl = final_long_model.predict_proba(X_live)[0][1]
    ps = final_short_model.predict_proba(X_live)[0][1]
    regime = lr['Regime'].values[0]
    atr = max(lr['ATR_Norm'].values[0], 0.01)
    fg = lr['Fear_Greed'].values[0] if 'Fear_Greed' in lr.columns else 50
    live_price = lr['Close'].values[0]

    sl_dist = atr * SL_MULT
    tp_dist = atr * TP_MULT

    tm_sig, col_tm = "CASH  ", BLU
    tm_sz, tm_prob = 0.0, max(pl, ps)
    tm_ex = ""

    if regime == 'Bull' and pl > LONG_THRESHOLD:
        tm_sig, col_tm, tm_sz, tm_prob = "LONG  ", GRN, min(
            RISK_PER_TRADE/sl_dist, 1.0), pl
        tm_ex = f"  |  SL: ${live_price*(1-sl_dist):,.0f}  TP: ${live_price*(1+tp_dist):,.0f}"
    elif regime == 'Bear' and ps > SHORT_THRESHOLD:
        tm_sig, col_tm, tm_sz, tm_prob = "SHORT ", RED, min(
            RISK_PER_TRADE/sl_dist, 1.0), ps
        tm_ex = f"  |  SL: ${live_price*(1+sl_dist):,.0f}  TP: ${live_price*(1-tp_dist):,.0f}"
    elif regime == 'Range':
        if pl > RANGE_THRESHOLD and pl > ps:
            tm_sig, col_tm, tm_sz, tm_prob = "LONG  ", GRN, min(
                RISK_PER_TRADE/sl_dist, 1.0), pl
            tm_ex = f"  |  SL: ${live_price*(1-sl_dist):,.0f}  TP: ${live_price*(1+tp_dist):,.0f}"
        elif ps > RANGE_THRESHOLD and ps > pl:
            tm_sig, col_tm, tm_sz, tm_prob = "SHORT ", RED, min(
                RISK_PER_TRADE/sl_dist, 1.0), ps
            tm_ex = f"  |  SL: ${live_price*(1+sl_dist):,.0f}  TP: ${live_price*(1-tp_dist):,.0f}"

    print("\n" + "═"*115)
    print(f"  {BLD}LIVE OUTLOOK / TOMORROW:{RST}  {col_tm}{BLD}{tm_sig.strip()}{RST}  | Prob: {tm_prob*100:.1f}%  Size: {tm_sz:.3f}x{tm_ex}")
    print(
        f"  Regime: {regime}  |  Fear/Greed: {fg:.0f}/100  |  P(Long): {pl*100:.1f}%  |  P(Short): {ps*100:.1f}%")
    print("═"*115)

    # ══════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════
    print("\n  Generating charts...")

    DARK = '#0d1117'
    PANEL = '#161b22'
    GRID = '#30363d'
    TEXT = '#e6edf3'
    DIMC = '#8b949e'
    GRNC = '#3fb950'
    REDC = '#f85149'
    BLUC = '#58a6ff'
    YELC = '#d29922'
    ORGC = '#fb8500'
    PURC = '#bc8cff'
    TEAL = '#39d353'

    def ax_style(ax, title, fs=9):
        ax.set_facecolor(PANEL)
        ax.spines[:].set_color(GRID)
        ax.tick_params(colors=DIMC, labelsize=7)
        ax.set_title(title, color=TEXT, fontsize=fs, fontweight='bold', pad=7)
        ax.grid(True, color=GRID, alpha=0.5, lw=0.5)
        ax.xaxis.label.set_color(DIMC)
        ax.yaxis.label.set_color(DIMC)

    # ─── FIGURE 1: PREDICTIVE PERFORMANCE & FINANCIALS ───────────────────────────
    fig1 = plt.figure(figsize=(22, 15), facecolor=DARK)
    gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.40, wspace=0.30)

    a1 = fig1.add_subplot(gs1[0, 0])
    ax_style(a1, f'① ROC Curve  (Long={auc_l:.3f}, Short={auc_s:.3f})')
    fpr_l, tpr_l, _ = roc_curve(trade_df['Label_Long'], trade_df['P_Long'])
    fpr_s, tpr_s, _ = roc_curve(trade_df['Label_Short'], trade_df['P_Short'])
    a1.plot(fpr_l, tpr_l, color=GRNC, lw=2, label=f'Long AUC={auc_l:.3f}')
    a1.plot(fpr_s, tpr_s, color=REDC, lw=2, label=f'Short AUC={auc_s:.3f}')
    a1.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--', label='Random')
    a1.set_xlabel('FPR')
    a1.set_ylabel('TPR')
    a1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    a2 = fig1.add_subplot(gs1[0, 1])
    ax_style(a2, '② Precision–Recall Curve')
    pr_l, re_l, _ = precision_recall_curve(
        trade_df['Label_Long'], trade_df['P_Long'])
    pr_s, re_s, _ = precision_recall_curve(
        trade_df['Label_Short'], trade_df['P_Short'])
    a2.plot(re_l, pr_l, color=GRNC, lw=2, label='Long Model')
    a2.plot(re_s, pr_s, color=REDC, lw=2, label='Short Model')
    a2.axvline(LONG_THRESHOLD, color=YELC, lw=1, ls=':',
               alpha=0.7, label=f'THR={LONG_THRESHOLD}')
    a2.set_xlabel('Recall')
    a2.set_ylabel('Precision')
    a2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    a3 = fig1.add_subplot(gs1[0, 2])
    ax_style(a3, '③ Signal Probability Distribution')
    bns = np.linspace(0, 1, 30)
    a3.hist(trade_df['P_Short'], bins=bns,
            alpha=0.5, color=REDC, label='P(Short)')
    a3.hist(trade_df['P_Long'], bins=bns,
            alpha=0.5, color=GRNC, label='P(Long)')
    a3.axvline(LONG_THRESHOLD, color=YELC, lw=2, ls='--', label='Trigger THR')
    a3.set_xlabel('Probability')
    a3.set_ylabel('Density')
    a3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    a4 = fig1.add_subplot(gs1[1, :])
    ax_style(a4, f'④ Strategy (net of fee) vs Buy & Hold — Last 180 Days')
    dx = df_180.index
    a4.plot(dx, cum_strat_180*100, color=GRNC, lw=2,
            label='Net Strategy', marker='o', ms=1.5)
    a4.plot(dx, cum_bh_180*100, color=BLUC, lw=2, label='Buy & Hold', ls='--')
    a4.axhline(0, color=GRID, lw=1)
    a4.fill_between(dx, cum_strat_180*100, 0, alpha=0.12, color=GRNC)
    a4.set_xlabel('Date')
    a4.set_ylabel('Return %')
    a4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

    a5 = fig1.add_subplot(gs1[2, :])
    ax_style(a5, f'⑤ Daily Net Returns & Active Confidence — Last 180 Days')
    bc5 = [GRNC if s == 1 else (REDC if s == -1 else GRID)
           for s in df_180['Signal']]
    a5.bar(dx, df_180['Strategy_Return']*100, color=bc5, alpha=0.75, width=1)
    a5.axhline(0, color=GRID, lw=1)
    a5b = a5.twinx()
    max_prob = df_180[['P_Long', 'P_Short']].max(axis=1) * 100
    a5b.plot(dx, max_prob, color=YELC, lw=1, alpha=0.6, label='Max Prob %')
    a5b.axhline(LONG_THRESHOLD*100, color=YELC, lw=0.8, ls='--', alpha=0.4)
    a5b.set_ylabel('Probability %', color=YELC, fontsize=7)
    a5b.tick_params(colors=YELC, labelsize=7)
    a5b.spines[:].set_color(GRID)
    a5b.set_ylim(0, 100)
    a5.set_xlabel('Date')
    a5.set_ylabel('Net Return %')
    pats = [Patch(color=GRNC, label='Long Executed'),
            Patch(color=REDC, label='Short Executed')]
    a5.legend(handles=pats, fontsize=7, facecolor=PANEL,
              labelcolor=TEXT, loc='upper left')

    fig1.suptitle(
        f'BTC-USD XGBOOST MACRO-QUANT  ·  PERFORMANCE & FINANCIALS  ·  {datetime.now().strftime("%Y-%m-%d")}\n'
        f'Long AUC: {auc_l:.3f} | Short AUC: {auc_s:.3f}  ·  Sharpe (180d): {sh_net_180:.2f} [{sh_net_lo_180:.2f},{sh_net_hi_180:.2f}]  ·  PF (180d): {pf_180:.2f}',
        color=TEXT, fontsize=10, fontweight='bold', y=0.97)

    plt.show()

    # ─── FIGURE 2: FEATURE INTELLIGENCE & DEEP RISK ─────────────────────────────
    fig2 = plt.figure(figsize=(20, 13), facecolor=DARK)
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.3)

    fl = (pd.DataFrame(imp_l_all).groupby('feature')['importance'].mean(
    ).reset_index().sort_values('importance', ascending=False)).head(15)
    fs = (pd.DataFrame(imp_s_all).groupby('feature')['importance'].mean(
    ).reset_index().sort_values('importance', ascending=False)).head(15)

    a6 = fig2.add_subplot(gs2[0, 0])
    ax_style(a6, '⑥ Feature Importance (LONG Model - Top 15)', 8.5)
    a6.barh(range(len(fl)), fl['importance'].values, color=GRNC, alpha=0.85)
    a6.set_yticks(range(len(fl)))
    a6.set_yticklabels(fl['feature'].values, fontsize=7.5, color=TEXT)
    a6.invert_yaxis()
    a6.set_xlabel("Average Gain / Weight")

    a7 = fig2.add_subplot(gs2[0, 1])
    ax_style(a7, '⑦ Feature Importance (SHORT Model - Top 15)', 8.5)
    a7.barh(range(len(fs)), fs['importance'].values, color=REDC, alpha=0.85)
    a7.set_yticks(range(len(fs)))
    a7.set_yticklabels(fs['feature'].values, fontsize=7.5, color=TEXT)
    a7.invert_yaxis()
    a7.set_xlabel("Average Gain / Weight")

    a8 = fig2.add_subplot(gs2[1, :])
    ax_style(
        a8, f'⑧ Drawdown Analysis (Last 180 Days)  ·  MaxDD {dd_180["max_dd"]*100:.1f}%  ·  Calmar {dd_180["calmar"]:.2f}', 8.5)
    wealth_s = (dd_180['wealth']-1)*100
    dd_s = dd_180['dd_series']*100
    a8.plot(dx, wealth_s, color=GRNC, lw=1.8, label='Strategy wealth')
    a8.fill_between(dx, dd_s, 0, alpha=0.45, color=REDC, label='Drawdown')
    a8.axhline(0, color=GRID, lw=1)
    a8b = a8.twinx()
    a8b.plot(dx, (dd_bh_180['wealth']-1)*100, color=BLUC,
             lw=1.5, ls='--', alpha=0.7, label='B&H wealth')
    a8b.set_ylabel('B&H Return %', color=BLUC, fontsize=7)
    a8b.tick_params(colors=BLUC, labelsize=7)
    a8b.spines[:].set_color(GRID)
    a8.set_xlabel('Date')
    a8.set_ylabel('Strategy Return / Drawdown %')
    mdd_i = int(np.argmin(dd_180['dd_series']))
    if len(dx) > 0 and mdd_i < len(dx):
        a8.annotate(f'MaxDD\n{dd_180["max_dd"]*100:.1f}%', xy=(dx[mdd_i], dd_s[mdd_i]), xytext=(dx[min(mdd_i+30, len(dx)-1)],
                    dd_s[mdd_i]-2.5), color=REDC, fontsize=7.5, fontweight='bold', arrowprops=dict(arrowstyle='->', color=REDC, lw=1.2))
    a8.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='upper left')
    a8b.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='lower right')

    fig2.suptitle(
        f'BTC-USD XGBOOST MACRO-QUANT  ·  FEATURE INTELLIGENCE & RISK  ·  {datetime.now().strftime("%Y-%m-%d")}',
        color=TEXT, fontsize=10, fontweight='bold', y=0.96)

    plt.show()
    print("  Done. ✓\n")


if __name__ == "__main__":
    main()
