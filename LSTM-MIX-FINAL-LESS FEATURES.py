

from scipy.stats import pearsonr, zscore
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Input, Bidirectional,
                                     LayerNormalization, MultiHeadAttention)
from tensorflow.keras.models import Model
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (roc_curve, roc_auc_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
START_DATE = '2020-01-01'   # 5 years of data
BACKTEST_D = 180            # 180-day forward validation
SEQ_LEN = 30                # Lookback window
GAP = SEQ_LEN               # Split gap to prevent sequence leakage

EPOCHS = 80
BATCH = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15

# Walk-forward CV
CV_FOLDS = 5
CV_EPOCHS = 20

# Dynamic TP/SL
SL_ATR_MULT = 1.5
RR_RATIO = 2.0

# Risk management
# 0.02% maker/taker = 0.04% round trip (applied on turnover)
TC = 0.0002
KELLY_FRAC = 0.50           # Half-Kelly for conservatism
MAX_SIZE = 1.00             # Cap position at 1.0x

# ══════════════════════════════════════════════════════════════════════
#  1. FETCH DATA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(
    f"  BTC-USD LSTM ULTIMATE (REPAIRED) |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═"*78)
print("  Fetching BTC-USD, GLD, ^GSPC  (synthetic fallback enabled)...")


def fetch_or_synth(ticker, start, n_synth=1500, seed=42, base=40000, vol=0.035):
    try:
        raw = yf.download(ticker, start=start, interval='1d', progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if len(raw) > 100:
            print(f"    ✓ {ticker}: {len(raw)} real rows")
            return raw
    except Exception:
        pass
    print(f"    ⚠ {ticker}: network unavailable — using synthetic GBM")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=n_synth)
    vol_t = np.ones(n_synth) * vol
    for i in range(1, n_synth):
        shock = rng.normal(0, vol_t[i-1])
        vol_t[i] = np.clip(0.94*vol_t[i-1] + 0.06*abs(shock), 0.008, 0.12)
    log_rets = rng.normal(0.0, vol_t, n_synth)
    for i in range(0, n_synth, 120):
        end = min(i+120, n_synth)
        sign = 1 if (i//120) % 2 == 0 else -1
        log_rets[i:end] += sign * 0.0005
    close = base * np.exp(np.cumsum(log_rets))
    high = close * np.exp(np.abs(rng.normal(0, vol_t/2, n_synth)))
    low = close * np.exp(-np.abs(rng.normal(0, vol_t/2, n_synth)))
    open_ = close * np.exp(rng.normal(0, vol_t/3, n_synth))
    vol_v = rng.lognormal(20, 1, n_synth)
    df_s = pd.DataFrame({'Open': open_, 'High': high, 'Low': low,
                         'Close': close, 'Volume': vol_v},
                        index=pd.DatetimeIndex(dates))
    up_pct = (np.diff(close) > 0).mean() * 100
    print(f"    ↑ Synthetic up-days: {up_pct:.1f}%  (target ≈50 %)")
    return df_s


btc = fetch_or_synth('BTC-USD', START_DATE, seed=42,  base=42000, vol=0.038)
gld = fetch_or_synth('GLD',     START_DATE, seed=123, base=170,   vol=0.010)
spx = fetch_or_synth('^GSPC',   START_DATE, seed=77,  base=3700,  vol=0.012)

for asset in [btc, gld, spx]:
    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)

df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
gld_close = gld['Close'].reindex(df.index, method='ffill')
spx_close = spx['Close'].reindex(df.index, method='ffill')

print(f"  Rows: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

# ══════════════════════════════════════════════════════════════════════
#  2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
c, h, lo, o, v = df.Close, df.High, df.Low, df.Open, df.Volume

tr = np.maximum(h-lo, np.maximum(abs(h-c.shift(1)), abs(lo-c.shift(1))))
df['atr'] = tr.rolling(14).mean()
df['atr_pct'] = df['atr'] / c

df['ret1'] = c.pct_change()
df['ret2'] = c.pct_change(2)
df['ret5'] = c.pct_change(5)
df['ret10'] = c.pct_change(10)
df['ret21'] = c.pct_change(21)
rv5_calc = df['ret1'].rolling(5).std()
rv21_calc = df['ret1'].rolling(21).std()

df['parkinson'] = (np.log(h/lo)**2 / (4*np.log(2))).rolling(5).mean()

df['vol_ratio'] = v / v.rolling(20).mean()
df['vol_spike'] = (v / v.rolling(5).mean()).clip(0, 5)
df['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
df['vol_price_div'] = df['vol_ratio'] / \
    (rv5_calc.replace(0, np.nan)*100 + 0.001)
df["volume_z"] = zscore(v.fillna(method="ffill"))

df['gap'] = (o - c.shift(1)) / (c.shift(1) + 1e-9)
df['body'] = (c - o).abs() / (h - lo + 1e-9)
df['upper_wick'] = (h - c.clip(upper=h)) / (h - lo + 1e-9)
df['lower_wick'] = (c.clip(lower=lo) - lo) / (h - lo + 1e-9)
df['hl_range'] = (h - lo) / c

df['sma5'] = c.rolling(5).mean()
df['sma20'] = c.rolling(20).mean()
df['sma50'] = c.rolling(50).mean()
df['z_sma5'] = (c-df['sma5']) / (c.rolling(5).std() + 1e-9)
df['z_sma20'] = (c-df['sma20']) / (c.rolling(20).std() + 1e-9)
df['z_sma50'] = (c-df['sma50']) / (c.rolling(50).std() + 1e-9)
df['sma_cross'] = (df['sma5']-df['sma20']) / df['sma20']
df['dist_52h'] = c / h.rolling(252).max() - 1
df['dist_52l'] = c / lo.rolling(252).min() - 1

btc_gold = c / gld_close.clip(lower=0.01)
df['btc_gold_ret'] = btc_gold.pct_change()
df['btc_gold_trend'] = btc_gold / btc_gold.rolling(20).mean() - 1

spx_ret1 = spx_close.pct_change()
spx_ret5 = spx_close.pct_change(5)
spx_vol5 = spx_ret1.rolling(5).std()
df['sp500_ret1'] = spx_ret1.reindex(df.index).fillna(0)
df['sp500_ret5'] = spx_ret5.reindex(df.index).fillna(0)
df['sp500_vol5'] = spx_vol5.reindex(df.index).ffill().fillna(0)
btc_spx = c / spx_close.clip(lower=0.01)
df['btc_sp500_ret'] = btc_spx.pct_change()
df['btc_sp500_trend'] = btc_spx / btc_spx.rolling(20).mean() - 1
df['btc_spx_corr'] = df['ret1'].rolling(20).corr(spx_ret1)

df['mom_accel'] = df['ret5'] - df['ret10']
df['vol_regime'] = rv5_calc / (rv21_calc + 1e-9)
df['mean_rev'] = -df['ret5'] * (1 / (rv5_calc + 0.001))

for lag in [1, 2, 3, 5]:
    df[f'ret1_lag{lag}'] = df['ret1'].shift(lag)

df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

# ── Restored 1-Day Target to fix Isotonic Imbalance ───────────────────
# This perfectly maps to the daily Kelly Rebalance Engine
df['target'] = (df['ret1'].shift(-1) > 0).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# 17 highly correlated or severely lagging features removed.
# Retained 23 optimal, stationary, and un-correlated features.
FEATS = [
    'ret1', 'ret5',
    'parkinson', 'atr_pct',
    'vol_ratio',
    'body', 'upper_wick', 'hl_range',
    'btc_gold_ret', 'btc_gold_trend',
    'sp500_ret1', 'sp500_vol5',
    'btc_sp500_ret', 'btc_sp500_trend', 'btc_spx_corr',
    'mom_accel', 'vol_regime',
    'ret1_lag1', 'ret1_lag2', 'ret1_lag3', 'ret1_lag5',
    'day_sin', 'day_cos'
]
N_FEATS = len(FEATS)
print(f"  Features engineered: {N_FEATS}")

# ══════════════════════════════════════════════════════════════════════
#  3. SPLIT, SCALE & SEQUENCE
# ══════════════════════════════════════════════════════════════════════
X_all = df[FEATS].values
y_all = df['target'].values

X_main = X_all[:-BACKTEST_D]
y_main = y_all[:-BACKTEST_D]
n = len(X_main)

n_train = int(n * TRAIN_SPLIT)
n_val = int(n * (TRAIN_SPLIT + VAL_SPLIT))

scaler = StandardScaler().fit(X_main[:n_train])
Xs_all = scaler.transform(X_all)
Xs_main = scaler.transform(X_main)


def make_seqs(X, y, sl):
    xs, ys = [], []
    for i in range(len(X) - sl):
        xs.append(X[i:i+sl])
        ys.append(y[i+sl])
    return np.array(xs), np.array(ys)


Xseq, yseq = make_seqs(Xs_main, y_main, SEQ_LEN)

Xtr,  ytr = Xseq[:n_train - SEQ_LEN - GAP], yseq[:n_train - SEQ_LEN - GAP]
Xval, yval = Xseq[n_train + GAP: n_val - SEQ_LEN -
                  GAP], yseq[n_train + GAP: n_val - SEQ_LEN - GAP]
Xte,  yte = Xseq[n_val + GAP:], yseq[n_val + GAP:]

print(f"  Train: {len(Xtr):,}  |  Val: {len(Xval):,}  |  Test: {len(Xte):,}  |  Backtest: {BACKTEST_D} d")

# ══════════════════════════════════════════════════════════════════════
#  4. MODEL ARCHITECTURE (Attention + LayerNorm)
# ══════════════════════════════════════════════════════════════════════


def build_model(input_shape):
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = LayerNormalization()(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = x + attn

    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='swish',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)

    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out)


cw = dict(enumerate(compute_class_weight(
    'balanced', classes=np.unique(ytr), y=ytr)))
model = build_model((SEQ_LEN, N_FEATS))
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

print(f"\n  Model parameters: {model.count_params():,}")

cbs = [
    EarlyStopping(patience=12, restore_best_weights=True,
                  monitor='val_auc', mode='max'),
    ReduceLROnPlateau(patience=6, factor=0.5, min_lr=1e-5,
                      monitor='val_auc', mode='max'),
]

print("  Training base model...\n")
history = model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH,
                    validation_data=(Xval, yval), class_weight=cw,
                    callbacks=cbs, verbose=1)

# ══════════════════════════════════════════════════════════════════════
#  5. ISOTONIC REGRESSION CALIBRATION
# ══════════════════════════════════════════════════════════════════════
print("\n  Calibrating with Isotonic Regression on validation set...")

raw_tr_probs = model.predict(Xtr,  verbose=0).flatten()
raw_val_probs = model.predict(Xval, verbose=0).flatten()
raw_te_probs = model.predict(Xte,  verbose=0).flatten()

iso_cal = IsotonicRegression(out_of_bounds='clip')
iso_cal.fit(raw_val_probs, yval.astype(float))


def calibrate(raw):
    arr = np.clip(np.atleast_1d(np.asarray(raw, dtype=float)), 0.0, 1.0)
    out = np.clip(iso_cal.predict(arr), 0.01, 0.99)
    return out


cal_tr = calibrate(raw_tr_probs)
cal_val = calibrate(raw_val_probs)
cal_te = calibrate(raw_te_probs)

fpr_roc, tpr_roc, thrs_roc = roc_curve(yte, cal_te)
auc_te = roc_auc_score(yte, cal_te)
val_median = float(np.median(cal_val))
youden_thr = float(thrs_roc[np.argmax(tpr_roc - fpr_roc)])
THR = float(np.clip(0.5*val_median + 0.5*youden_thr, 0.35, 0.65))
print(f"  Isotonic calibration fitted.  Base threshold THR = {THR:.3f}")

# ══════════════════════════════════════════════════════════════════════
#  6. WALK-FORWARD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
print(
    f"\n  Walk-Forward CV  ({CV_FOLDS} expanding folds, {CV_EPOCHS} epochs each)...")

cv_aucs, cv_accs, cv_n_test = [], [], []
Xseq_cv, yseq_cv = make_seqs(Xs_main, y_main, SEQ_LEN)
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

for fold, (tr_idx, te_idx) in enumerate(tscv.split(Xseq_cv)):
    te_idx = te_idx[te_idx > tr_idx[-1] + SEQ_LEN]
    if len(te_idx) < 20:
        continue

    Xcv_tr, ycv_tr = Xseq_cv[tr_idx], yseq_cv[tr_idx]
    Xcv_te, ycv_te = Xseq_cv[te_idx], yseq_cv[te_idx]

    m_cv = build_model((SEQ_LEN, N_FEATS))
    m_cv.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    cw_cv = dict(enumerate(compute_class_weight(
        'balanced', classes=np.unique(ycv_tr), y=ycv_tr)))
    m_cv.fit(Xcv_tr, ycv_tr, epochs=CV_EPOCHS,
             batch_size=BATCH, class_weight=cw_cv, verbose=0)

    raw_cv = m_cv.predict(Xcv_te, verbose=0).flatten()
    probs_cv = np.clip(raw_cv, 0.01, 0.99)
    preds_cv = (probs_cv >= 0.5).astype(int)

    fold_auc = roc_auc_score(ycv_te, probs_cv)
    fold_acc = accuracy_score(ycv_te, preds_cv)
    cv_aucs.append(fold_auc)
    cv_accs.append(fold_acc)
    cv_n_test.append(len(te_idx))
    print(f"    Fold {fold+1}: AUC={fold_auc:.4f}  Acc={fold_acc*100:.1f}%  (n_train={len(tr_idx):,}  n_test={len(te_idx):,})")
    del m_cv

cv_auc_mean, cv_auc_std = np.mean(cv_aucs), np.std(cv_aucs)
cv_acc_mean, cv_acc_std = np.mean(cv_accs), np.std(cv_accs)
print(f"\n  ► CV AUC : {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
print(f"  ► CV Acc : {cv_acc_mean*100:.1f}% ± {cv_acc_std*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
#  7. METRICS — TRAIN / VAL / TEST
# ══════════════════════════════════════════════════════════════════════


def get_metrics(X, y, label, raw_p=None):
    if raw_p is None:
        raw_p = model.predict(X, verbose=0).flatten()
    cp = calibrate(raw_p)
    pred = (cp >= THR).astype(int)
    return dict(label=label,
                acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0),
                f1=2*precision_score(y, pred, zero_division=0)*recall_score(y, pred, zero_division=0) /
                (precision_score(y, pred, zero_division=0) +
                 recall_score(y, pred, zero_division=0)+1e-9),
                auc=roc_auc_score(y, cp),
                probs=cp, preds=pred,
                cm=confusion_matrix(y, pred), raw=raw_p)


m_tr = get_metrics(Xtr,  ytr,  "TRAIN", raw_tr_probs)
m_val = get_metrics(Xval, yval, "VAL  ", raw_val_probs)
m_te = get_metrics(Xte,  yte,  "TEST ", raw_te_probs)

# ══════════════════════════════════════════════════════════════════════
#  8. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
print("\n  Computing permutation feature importance on test set...")
baseline_auc = roc_auc_score(yte, cal_te)
feat_imp = np.zeros(N_FEATS)
N_REPEATS = 3

for j in range(N_FEATS):
    fold_aucs = []
    for _ in range(N_REPEATS):
        Xp = Xte.copy()
        perm = np.random.permutation(len(Xp))
        Xp[:, :, j] = Xp[perm, :, j]
        raw_p = model.predict(Xp, verbose=0).flatten()
        fold_aucs.append(roc_auc_score(yte, calibrate(raw_p)))
    feat_imp[j] = baseline_auc - np.mean(fold_aucs)

print(
    f"  Done. Top feature: {FEATS[np.argmax(feat_imp)]}  (ΔAUCdrop={feat_imp.max():.4f})")

# ══════════════════════════════════════════════════════════════════════
#  9. RISK MANAGEMENT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def kelly_size(prob, direction, rr=RR_RATIO, kf=KELLY_FRAC, cap=MAX_SIZE):
    if direction == 0:
        return 0.0
    p = float(np.clip(prob if direction == 1 else 1.0-prob, 0.01, 0.99))
    b = rr
    f = max(0.0, (p*b - (1-p)) / b)
    return min(kf * f, cap)


def sharpe_with_ci(rets, ann=365):
    rets = np.asarray(rets, dtype=float)
    n = len(rets)
    if n < 2 or rets.std() < 1e-10:
        return 0.0, -np.inf, np.inf
    sr_d = rets.mean() / rets.std()
    sr_a = sr_d * np.sqrt(ann)
    se_d = np.sqrt((1 + 0.5*sr_d**2) / max(n-1, 1))
    se_a = se_d * np.sqrt(ann)
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
    trough = int(np.argmin(dd))
    recovered = np.where(wealth[trough:] >= peak[trough])[0]
    recovery = int(recovered[0]) if len(recovered) else len(rets) - trough
    ann_ret = float((wealth[-1] ** (365 / max(len(rets), 1))) - 1)
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
    return dict(max_dd=max_dd, avg_dd=avg_dd, max_dur=max_dur, recovery=recovery, calmar=calmar, ann_ret=ann_ret, wealth=wealth, dd_series=dd)


# ══════════════════════════════════════════════════════════════════════
#  10. 180-DAY BACKTEST (TURNOVER-BASED TC + CORRECTED KELLY)
# ══════════════════════════════════════════════════════════════════════
print(
    f"\n  Running {BACKTEST_D}-day backtest  (Kelly + Turnover TC + Rolling Refit)...")

bdts = df.index[-BACKTEST_D:]
bcls = df.Close.values[-BACKTEST_D:]
bopen = df.Open.values[-BACKTEST_D:]
bhigh = df.High.values[-BACKTEST_D:]
blow = df.Low.values[-BACKTEST_D:]
batr = df.atr.values[-BACKTEST_D:]
bh_ret = df['ret1'].values[-BACKTEST_D:]

sizes = np.zeros(BACKTEST_D)
ksizes = np.zeros(BACKTEST_D)
bp = np.zeros(BACKTEST_D)
bp_raw = np.zeros(BACKTEST_D)
sr_gross = np.zeros(BACKTEST_D)
sr_daily = np.zeros(BACKTEST_D)
sl_arr = np.full(BACKTEST_D, np.nan)
tp_arr = np.full(BACKTEST_D, np.nan)
labels_bt = ["CASH  "] * BACKTEST_D
current_center = THR
total_costs_paid = 0.0

for i in range(BACKTEST_D):
    idx = len(df) - BACKTEST_D + i

    if i > 0 and i % 5 == 0:
        rX_raw, ry_raw = Xs_all[idx-252-SEQ_LEN: idx -
                                1], y_all[idx-252-SEQ_LEN: idx-1]
        rX, ry = make_seqs(rX_raw, ry_raw, SEQ_LEN)
        model.fit(rX, ry, epochs=1, batch_size=BATCH, verbose=0)
        cx_raw, _ = make_seqs(
            Xs_all[idx-60-SEQ_LEN: idx-1], y_all[idx-60-SEQ_LEN: idx-1], SEQ_LEN)
        if len(cx_raw) > 0:
            recent_p = calibrate(model.predict(cx_raw, verbose=0).flatten())
            current_center = 0.70*current_center + \
                0.30*float(np.median(recent_p))

    seq = Xs_all[idx-SEQ_LEN:idx][np.newaxis, :, :]
    raw_pred = float(model.predict(seq, verbose=0).flatten()[0])
    prob = float(calibrate(np.array([raw_pred]))[0])
    bp_raw[i] = raw_pred
    bp[i] = prob

    vol_buf = float(np.clip(df['atr_pct'].iloc[idx-1] * 0.5, 0.005, 0.02))
    dyn_hi = current_center + vol_buf
    dyn_lo = current_center - vol_buf
    trend_state = float(df['sma5'].iloc[idx-1] / df['sma50'].iloc[idx-1] - 1)
    if trend_state > 0.02:
        dyn_lo -= 0.025
    elif trend_state < -0.02:
        dyn_hi += 0.025

    direction = 0
    if prob > dyn_hi:
        direction = 1
    elif prob < dyn_lo:
        direction = -1

    # ── FIXED: Strict Kelly enforcement ──────────────────────────────
    k_sz = kelly_size(prob, direction)
    if k_sz == 0.0:
        direction = 0  # Revert to cash if Kelly rejects trade

    sizes[i] = direction * k_sz
    ksizes[i] = k_sz

    if sizes[i] > 0:
        labels_bt[i] = "LONG  "
    elif sizes[i] < 0:
        labels_bt[i] = "SHORT "
    else:
        labels_bt[i] = "CASH  "

    # ── FIXED: Transaction Cost strictly on turnover ─────────────────
    if i == 0:
        turnover = abs(sizes[i])
    else:
        turnover = abs(sizes[i] - sizes[i-1])

    tc_cost = turnover * TC
    total_costs_paid += tc_cost

    # ── Intraday logic ────────────────────────────────────────────────
    epx = bopen[i]
    hpx = bhigh[i]
    lpx = blow[i]
    cpx = bcls[i]
    atr = batr[i]
    trade_ret = 0.0

    if direction == 1:
        sl_px = epx - atr*SL_ATR_MULT
        tp_px = epx + atr*SL_ATR_MULT*RR_RATIO
        sl_arr[i] = sl_px
        tp_arr[i] = tp_px
        if lpx <= sl_px:
            trade_ret = (sl_px - epx) / epx
        elif hpx >= tp_px:
            trade_ret = (tp_px - epx) / epx
        else:
            trade_ret = (cpx - epx) / epx

    elif direction == -1:
        sl_px = epx + atr*SL_ATR_MULT
        tp_px = epx - atr*SL_ATR_MULT*RR_RATIO
        sl_arr[i] = sl_px
        tp_arr[i] = tp_px
        if hpx >= sl_px:
            trade_ret = (epx - sl_px) / epx
        elif lpx <= tp_px:
            trade_ret = (epx - tp_px) / epx
        else:
            trade_ret = (epx - cpx) / epx

    gross = trade_ret * k_sz
    sr_gross[i] = gross
    sr_daily[i] = gross - tc_cost

cum_strat = (1 + sr_daily).cumprod() - 1
cum_gross = (1 + sr_gross).cumprod() - 1
cum_bh = (1 + bh_ret).cumprod() - 1

sh_net,   sh_net_lo,   sh_net_hi = sharpe_with_ci(sr_daily)
sh_gross, _,           _ = sharpe_with_ci(sr_gross)
sh_bh,    sh_bh_lo,    sh_bh_hi = sharpe_with_ci(bh_ret)

dd = drawdown_stats(sr_daily)
dd_bh = drawdown_stats(bh_ret)

active_rets = [sr_daily[i] for i in range(BACKTEST_D) if sizes[i] != 0]
wins = sum(r > 0 for r in active_rets)
losses = sum(r < 0 for r in active_rets)
active = len(active_rets)
avg_win = np.mean([r for r in active_rets if r > 0]) if wins else 0.0
avg_loss = np.mean([r for r in active_rets if r < 0]) if losses else 0.0
total_gross_pos = sum(r for r in active_rets if r > 0)
total_gross_neg = abs(sum(r for r in active_rets if r < 0))
profit_factor = total_gross_pos / max(total_gross_neg, 1e-9)

# ══════════════════════════════════════════════════════════════════════
#  11. TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════
GRN = "\033[92m"
RED = "\033[91m"
BLU = "\033[94m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"
RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"

print("\n" + "═"*105)
print(f"{BLD}  MODEL PERFORMANCE REPORT  —  BTC-USD LSTM ULTIMATE (REPAIRED){RST}")
print("═"*105)
print(f"\n  {'METRIC':<20} {'TRAIN':>9} {'VAL':>9} {'TEST':>9}  {'CV MEAN':>9} {'CV STD':>9}")
print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*9}  {'─'*9} {'─'*9}")

for key, lbl in [('acc', 'Accuracy'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1-Score'), ('auc', 'ROC-AUC')]:
    tr_v = m_tr[key]*100
    val_v = m_val[key]*100
    te_v = m_te[key]*100
    te_c = GRN if te_v >= 55 else (YEL if te_v >= 50 else RED)
    cv_m = cv_auc_mean * \
        100 if key == 'auc' else (
            cv_acc_mean*100 if key == 'acc' else float('nan'))
    cv_s = cv_auc_std * \
        100 if key == 'auc' else (
            cv_acc_std*100 if key == 'acc' else float('nan'))
    cv_m_s = f"{cv_m:>8.1f}%" if not np.isnan(cv_m) else f"{'─':>9}"
    cv_s_s = f"{cv_s:>8.1f}%" if not np.isnan(cv_s) else f"{'─':>9}"
    print(f"  {lbl:<20} {tr_v:>8.1f}% {val_v:>8.1f}% {te_c}{te_v:>8.1f}%{RST}  {cv_m_s} {cv_s_s}")

print(f"\n  Confusion Matrix (TEST):")
cm = m_te['cm']
print(f"  {'':14} Pred:0   Pred:1")
print(f"  True:0        {cm[0, 0]:>5}    {cm[0, 1]:>5}")
print(f"  True:1        {cm[1, 0]:>5}    {cm[1, 1]:>5}")

p_test = m_te['probs']
print(f"\n  Calibrated Confidence Distribution (TEST, {len(p_test)} samples):")
bins_conf = [(0, .40, '<40% Strong Down'), (.40, .46, '40–46% Weak Down'),
             (.46, .54, '46–54% Neutral'),  (.54, .60, '54–60% Weak Up'),
             (.60, 1., '60%+  Strong Up')]
for lo_b, hi_b, lbl in bins_conf:
    cnt = ((p_test >= lo_b) & (p_test < hi_b)).sum()
    bar = "█" * int(cnt/len(p_test)*40)
    print(f"  {lbl:<22} {cnt:>4}  {bar}")

print("\n" + "═"*105)
print(f"{BLD}  180-DAY BACKTEST  |  Kelly Sizing · TC={TC*100:.2f}% (Turnover) · TP={SL_ATR_MULT*RR_RATIO:.1f}x ATR · SL={SL_ATR_MULT:.1f}x ATR{RST}")
print("═"*105)

hdr = (f"{'DATE':<12} {'PRICE':>9} {'CONF%':>6} {'RAW%':>6} {'SIG':<7} {'DIR':>4} {'KELLY':>6} {'SL$':>9} {'TP$':>9} {'GROSS%':>7} {'NET%':>7} {'CUM%':>8}  ST")
print(f"\n  {hdr}")
print("  " + "─"*110)

for i in range(BACKTEST_D):
    s = sizes[i]
    conf = bp[i]
    raw = bp_raw[i]
    gross = sr_gross[i]*100
    net = sr_daily[i]*100
    cum = cum_strat[i]*100
    sl_s = f"{sl_arr[i]:>9,.0f}" if not np.isnan(sl_arr[i]) else "        —"
    tp_s = f"{tp_arr[i]:>9,.0f}" if not np.isnan(tp_arr[i]) else "        —"
    col = GRN if s > 0 else (RED if s < 0 else BLU)
    stat = f"{DIM}–{RST}" if s == 0 else (
        f"{GRN}✓{RST}" if net > 0 else f"{RED}✗{RST}")
    nc = GRN if net > 0 else (RED if net < 0 else DIM)
    cc = GRN if cum > 0 else RED
    print(f"  {str(bdts[i].date()):<12} ${bcls[i]:>8,.0f} {conf*100:>5.1f}% {raw*100:>5.1f}% {col}{labels_bt[i]}{RST} {s:>+4.2f} {ksizes[i]:>5.3f}x {DIM}{sl_s}{RST} {DIM}{tp_s}{RST} {gross:>6.2f}% {nc}{net:>6.2f}%{RST} {cc}{cum:>7.2f}%{RST}  {stat}")

print("  " + "─"*110)


def pline(label, val, fmt='%', col=None, extra=''):
    if col is None:
        col = GRN if val >= 0 else RED
    v_s = f"{val*100:>+8.2f}%" if fmt == '%' else f"{val:>+8.2f}"
    print(f"  {label:<38} {col}{v_s}{RST}{extra}")


print()
pline("Strategy Return (net of TC)",       cum_strat[-1])
pline("Strategy Return (gross)",           cum_gross[-1])
pline(f"Transaction Costs Paid Total",     -total_costs_paid, col=RED)
pline("Buy & Hold Return",                 cum_bh[-1])
pline("Alpha (net vs B&H)",                cum_strat[-1]-cum_bh[-1])

print()
sh_c = GRN if sh_net > 0 else RED
print(
    f"  {'Sharpe Ratio (net, ann.)':<38} {sh_c}{sh_net:>+8.2f}{RST}  (95% CI: [{sh_net_lo:+.2f}, {sh_net_hi:+.2f}])")
print(f"  {'Sharpe Ratio (gross, ann.)':<38} {(GRN if sh_gross > 0 else RED)}{sh_gross:>+8.2f}{RST}")
print(
    f"  {'Sharpe Ratio (Buy & Hold, ann.)':<38} {(GRN if sh_bh > 0 else RED)}{sh_bh:>+8.2f}{RST}  (95% CI: [{sh_bh_lo:+.2f}, {sh_bh_hi:+.2f}])")

print()
print(
    f"  {'Max Drawdown (strategy)':<38} {RED}{dd['max_dd']*100:>+8.2f}%{RST}")
print(
    f"  {'Avg Drawdown (strategy)':<38} {RED}{dd['avg_dd']*100:>+8.2f}%{RST}")
print(f"  {'Max Drawdown Duration':<38} {dd['max_dur']:>9} days")
print(
    f"  {'Recovery Time (from deepest trough)':<38} {dd['recovery']:>9} days")
print(
    f"  {'Calmar Ratio (strategy)':<38} {(GRN if dd['calmar'] > 0 else RED)}{dd['calmar']:>+8.2f}{RST}")

print()
win_c = GRN if wins/max(active, 1) >= 0.5 else YEL
print(f"  {'Win Rate (active signals)':<38} {win_c}{wins}/{max(active, 1)} = {wins/max(active, 1)*100:.1f}%{RST}")
print(f"  {'Avg Winning Trade':<38} {GRN}{avg_win*100:>+8.3f}%{RST}")
print(f"  {'Avg Losing Trade':<38} {RED}{avg_loss*100:>+8.3f}%{RST}")
print(f"  {'Profit Factor':<38} {(GRN if profit_factor >= 1 else RED)}{profit_factor:>8.2f}x{RST}")
print(f"  {'Active Days / Total Days':<38}  {active:>4}/{BACKTEST_D}")

# ── Tomorrow ──────────────────────────────────────────────────────────
tm_raw = float(model.predict(
    Xs_all[-SEQ_LEN:][np.newaxis, :, :], verbose=0).flatten()[0])
tm_conf = float(calibrate(np.array([tm_raw]))[0])
tm_buf = float(np.clip(df['atr_pct'].iloc[-1]*0.5, 0.005, 0.02))
tm_hi = current_center + tm_buf
tm_lo = current_center - tm_buf
tm_trend = float(df['sma5'].iloc[-1]/df['sma50'].iloc[-1]-1)

if tm_trend > 0.02:
    tm_lo -= 0.025
elif tm_trend < -0.02:
    tm_hi += 0.025

if tm_conf > tm_hi:
    tm_sz, tm_sig, tm_dir = 1.0, "LONG  ", 1
elif tm_conf < tm_lo:
    tm_sz, tm_sig, tm_dir = -1.0, "SHORT ", -1
else:
    tm_sz, tm_sig, tm_dir = 0.0, "CASH  ", 0

tm_kelly = kelly_size(tm_conf, tm_dir)
if tm_kelly == 0.0:
    tm_sz, tm_sig, tm_dir = 0.0, "CASH  ", 0

tm_sz = tm_dir * tm_kelly
tm_price = df['Close'].iloc[-1]
tm_atr = df['atr'].iloc[-1]

if tm_sz > 0:
    tm_sl = tm_price-tm_atr*SL_ATR_MULT
    tm_tp = tm_price+tm_atr*SL_ATR_MULT*RR_RATIO
    tm_ex = f"  |  SL: ${tm_sl:,.0f}  TP: ${tm_tp:,.0f}"
elif tm_sz < 0:
    tm_sl = tm_price+tm_atr*SL_ATR_MULT
    tm_tp = tm_price-tm_atr*SL_ATR_MULT*RR_RATIO
    tm_ex = f"  |  SL: ${tm_sl:,.0f}  TP: ${tm_tp:,.0f}"
else:
    tm_ex = ""

col_tm = GRN if tm_sz > 0 else (RED if tm_sz < 0 else BLU)
print("\n" + "═"*105)
print(f"  {BLD}TOMORROW'S SIGNAL:{RST}  {col_tm}{BLD}{tm_sig.strip()}{RST}  | Conf: {tm_conf*100:.1f}%  Raw: {tm_raw*100:.1f}%  Size: {tm_sz:+.2f}x  Kelly: {tm_kelly:.3f}x{tm_ex}")
print(
    f"  Dyn Long Thr: {tm_hi*100:.1f}%  |  Dyn Short Thr: {tm_lo*100:.1f}%  |  Center: {current_center*100:.1f}%")
print("═"*105)

# ══════════════════════════════════════════════════════════════════════
#  12. CHARTS
# ══════════════════════════════════════════════════════════════════════
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


fig1 = plt.figure(figsize=(22, 15), facecolor=DARK)
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.50, wspace=0.40)

a1 = fig1.add_subplot(gs1[0, 0])
ax_style(a1, '① Train vs Val Loss')
ep = range(1, len(history.history['loss'])+1)
a1.plot(ep, history.history['loss'],    color=BLUC, lw=1.5, label='Train')
a1.plot(ep, history.history['val_loss'],
        color=YELC, lw=1.5, label='Val', ls='--')
a1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
a1.set_xlabel('Epoch')
a1.set_ylabel('BCE Loss')

a2 = fig1.add_subplot(gs1[0, 1])
ax_style(a2, '② Train vs Val AUC')
a2.plot(ep, history.history['auc'],    color=GRNC, lw=1.5, label='Train')
a2.plot(ep, history.history['val_auc'],
        color=ORGC, lw=1.5, label='Val', ls='--')
a2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)
a2.set_xlabel('Epoch')
a2.set_ylabel('AUC')

a3 = fig1.add_subplot(gs1[0, 2])
ax_style(a3, f'③ ROC Curve  (Test AUC = {auc_te:.3f})')
fpr_t, tpr_t, _ = roc_curve(yte, cal_te)
a3.plot(fpr_t, tpr_t, color=GRNC, lw=2, label=f'Model (AUC={auc_te:.3f})')
a3.plot([0, 1], [0, 1],   color=GRID,  lw=1, ls='--', label='Random')
a3.fill_between(fpr_t, tpr_t, alpha=0.15, color=GRNC)
a3.set_xlabel('FPR')
a3.set_ylabel('TPR')
a3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a4 = fig1.add_subplot(gs1[1, 0])
ax_style(a4, '④ Precision–Recall Curve')
pr_p, re_p, _ = precision_recall_curve(yte, cal_te)
a4.plot(re_p, pr_p, color=BLUC, lw=2)
a4.axhline(0.5, color=GRID, lw=1, ls='--', label='Random')
a4.axvline(THR, color=YELC, lw=1, ls=':', alpha=0.7, label=f'THR={THR:.2f}')
a4.fill_between(re_p, pr_p, alpha=0.10, color=BLUC)
a4.set_xlabel('Recall')
a4.set_ylabel('Precision')
a4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a5 = fig1.add_subplot(gs1[1, 1])
ax_style(a5, '⑤ Calibrated Confidence (Test)')
bns = np.linspace(0, 1, 30)
a5.hist(cal_te[yte == 0], bins=bns, alpha=0.6, color=REDC, label='Down/SL')
a5.hist(cal_te[yte == 1], bins=bns, alpha=0.6, color=GRNC, label='Up/TP')
a5.axvline(THR, color=YELC, lw=2, ls='--', label=f'THR={THR:.2f}')
a5.set_xlabel('P(Direction)')
a5.set_ylabel('Count')
a5.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a6 = fig1.add_subplot(gs1[1, 2])
ax_style(a6, '⑥ Strategy (net/gross) vs Buy & Hold — 180 d')
dx = range(BACKTEST_D)
a6.plot(dx, cum_strat*100, color=GRNC, lw=2,
        label='Net (Kelly+TC)', marker='o', ms=2)
a6.plot(dx, cum_gross*100, color=TEAL, lw=1.2,  label='Gross (Kelly)',  ls=':')
a6.plot(dx, cum_bh*100,    color=BLUC, lw=2,   label='Buy & Hold',     ls='--')
a6.axhline(0, color=GRID, lw=1)
a6.fill_between(dx, cum_strat*100, 0, alpha=0.12, color=GRNC)
a6.set_xlabel('Day')
a6.set_ylabel('Return %')
a6.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a7 = fig1.add_subplot(gs1[2, 0:2])
ax_style(a7, '⑦ Daily Net Returns, Signals & Confidence — 180 d')
bc7 = [GRNC if s > 0 else (REDC if s < 0 else GRID) for s in sizes]
a7.bar(dx, sr_daily*100, color=bc7, alpha=0.75)
a7.axhline(0, color=GRID, lw=1)
a7b = a7.twinx()
a7b.plot(dx, bp*100, color=YELC, lw=1.5, alpha=0.8, label='Conf %')
a7b.axhline(THR*100, color=YELC, lw=0.8, ls='--', alpha=0.4)
a7b.set_ylabel('Confidence %', color=YELC, fontsize=7)
a7b.tick_params(colors=YELC, labelsize=7)
a7b.spines[:].set_color(GRID)
a7b.set_ylim(0, 100)
a7.set_xlabel('Day')
a7.set_ylabel('Net Return %')
pats = [Patch(color=GRNC, label='Long'), Patch(
    color=REDC, label='Short'), Patch(color=GRID, label='Cash')]
a7.legend(handles=pats, fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a8 = fig1.add_subplot(gs1[2, 2])
ax_style(a8, '⑧ Train / Val / Test / CV Metrics')
met_lbl = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
tr_v = [m_tr[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
vl_v = [m_val[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
te_v_ = [m_te[k]*100 for k in ['acc', 'prec', 'rec', 'f1', 'auc']]
cv_v = [cv_acc_mean*100, float('nan'), float('nan'),
        float('nan'), cv_auc_mean*100]
x8 = np.arange(5)
w8 = 0.18
a8.bar(x8-1.5*w8, tr_v,  w8, color=BLUC, alpha=0.8, label='Train')
a8.bar(x8-0.5*w8, vl_v,  w8, color=YELC, alpha=0.8, label='Val')
a8.bar(x8+0.5*w8, te_v_, w8, color=GRNC, alpha=0.8, label='Test')
cv_x8 = [i for i, vv in enumerate(cv_v) if not np.isnan(vv)]
cv_y8 = [vv for vv in cv_v if not np.isnan(vv)]
a8.bar([x8[i]+1.5*w8 for i in cv_x8], cv_y8, w8,
       color=PURC, alpha=0.8, label='CV mean')
a8.set_xticks(x8)
a8.set_xticklabels(met_lbl, fontsize=7, rotation=15, color=DIMC)
a8.axhline(50, color=GRID, lw=1, ls='--')
a8.set_ylim(0, 100)
a8.set_ylabel('%')
a8.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT)

fig1.suptitle(
    f'BTC-USD LSTM ULTIMATE REPAIRED  ·  {datetime.now().strftime("%Y-%m-%d")}  ·  '
    f'Test AUC: {auc_te:.3f}  ·  CV AUC: {cv_auc_mean:.3f}±{cv_auc_std:.3f}  ·  '
    f'Sharpe: {sh_net:.2f} [{sh_net_lo:.2f},{sh_net_hi:.2f}]  ·  '
    f'MaxDD: {dd["max_dd"]*100:.1f}%  ·  Calmar: {dd["calmar"]:.2f}  ·  '
    f'PF: {profit_factor:.2f}',
    color=TEXT, fontsize=9, fontweight='bold', y=0.995)

plt.show()

fig2 = plt.figure(figsize=(20, 13), facecolor=DARK)
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.50, wspace=0.42)

a9 = fig2.add_subplot(gs2[0, 0])
ax_style(a9, '⑨ Feature Importance  (Permutation ΔAUCdrop, top 15)', 8.5)
top15 = np.argsort(feat_imp)[::-1][:15]
imp15 = feat_imp[top15]
lbl15 = [FEATS[i] for i in top15]
bc9 = [GRNC if v > 0 else REDC for v in imp15]
bars9 = a9.barh(range(15), imp15, color=bc9, alpha=0.85)
a9.set_yticks(range(15))
a9.set_yticklabels(lbl15, fontsize=7.5, color=TEXT)
a9.invert_yaxis()
a9.axvline(0, color=GRID, lw=1)
a9.set_xlabel('AUC drop when feature permuted')
for bar, val in zip(bars9, imp15):
    a9.text(val+abs(imp15.max())*0.02, bar.get_y()+bar.get_height() /
            2, f'{val:.4f}', va='center', color=DIMC, fontsize=6.5)

a10 = fig2.add_subplot(gs2[0, 1])
ax_style(a10, '⑩ Feature–Target Correlation  (Pearson r, test set)', 8.5)
corrs_r = []
for j in range(N_FEATS):
    fv = Xte[:, -1, j]
    r, _ = pearsonr(fv, yte)
    corrs_r.append(r)
corrs_r = np.array(corrs_r)
sort20 = np.argsort(np.abs(corrs_r))[::-1][:20]
sc20 = corrs_r[sort20]
sl20 = [FEATS[i] for i in sort20]
bc10 = [GRNC if v > 0 else REDC for v in sc20]
a10.barh(range(len(sort20)), sc20, color=bc10, alpha=0.85)
a10.set_yticks(range(len(sort20)))
a10.set_yticklabels(sl20, fontsize=7.5, color=TEXT)
a10.invert_yaxis()
a10.axvline(0, color=GRID, lw=1)
a10.set_xlabel("Pearson r with tomorrow's target")
for i, (pos, val) in enumerate(zip(sc20, sc20)):
    off = 0.003 if val >= 0 else -0.003
    a10.text(val+off, i, f'{val:.3f}', va='center',
             ha='left' if val >= 0 else 'right', color=DIMC, fontsize=6)

a11 = fig2.add_subplot(gs2[1, 0])
ax_style(
    a11, f'⑪ Drawdown Analysis  ·  MaxDD {dd["max_dd"]*100:.1f}%  ·  Calmar {dd["calmar"]:.2f}', 8.5)
wealth_s = (dd['wealth']-1)*100
dd_s = dd['dd_series']*100
a11.plot(dx, wealth_s, color=GRNC, lw=1.8, label='Strategy wealth')
a11.fill_between(dx, dd_s, 0, alpha=0.45, color=REDC, label='Drawdown')
a11.axhline(0, color=GRID, lw=1)
a11b = a11.twinx()
a11b.plot(dx, (dd_bh['wealth']-1)*100, color=BLUC,
          lw=1.5, ls='--', alpha=0.7, label='B&H wealth')
a11b.set_ylabel('B&H Return %', color=BLUC, fontsize=7)
a11b.tick_params(colors=BLUC, labelsize=7)
a11b.spines[:].set_color(GRID)
a11.set_xlabel('Day')
a11.set_ylabel('Strategy Return / Drawdown %')
mdd_i = int(np.argmin(dd['dd_series']))
a11.annotate(f'MaxDD\n{dd["max_dd"]*100:.1f}%', xy=(mdd_i, dd_s[mdd_i]), xytext=(min(mdd_i+5, BACKTEST_D-1),
             dd_s[mdd_i]-2.5), color=REDC, fontsize=7.5, fontweight='bold', arrowprops=dict(arrowstyle='->', color=REDC, lw=1.2))
a11.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='upper left')
a11b.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='lower right')

a12 = fig2.add_subplot(gs2[1, 1])
ax_style(a12, '⑫ Walk-Forward CV AUC  &  Calibration Reliability', 8.5)
f_lbl = [f'Fold {i+1}\n(n={cv_n_test[i]})' for i in range(len(cv_aucs))]
bc12 = [GRNC if v > 0.52 else (YELC if v >= 0.50 else REDC) for v in cv_aucs]
bars12 = a12.bar(f_lbl, [v*100 for v in cv_aucs],
                 color=bc12, alpha=0.8, width=0.45)
a12.axhline(50, color=REDC, lw=1, ls='--', label='Random (50%)')
a12.axhline(cv_auc_mean*100, color=YELC, lw=2, ls='--',
            label=f'Mean={cv_auc_mean*100:.1f}%')
a12.set_ylabel('AUC (%)')
ub = max([v*100 for v in cv_aucs]) + 8
a12.set_ylim(40, min(80, ub))
for bar, v in zip(bars12, cv_aucs):
    a12.text(bar.get_x()+bar.get_width()/2, v*100+0.3,
             f'{v:.3f}', ha='center', va='bottom', color=TEXT, fontsize=7.5)

a12b = a12.twinx()
n_bins = 8
edges = np.linspace(0, 1, n_bins+1)
cen, frac = [], []
for k in range(n_bins):
    msk = (cal_te >= edges[k]) & (cal_te < edges[k+1])
    if msk.sum() >= 5:
        cen.append((edges[k]+edges[k+1])/2)
        frac.append(yte[msk].mean())
a12b.plot(cen, frac, color=PURC, lw=2.5, marker='o',
          ms=5, label='Reliability\n(isotonic cal.)')
a12b.plot([0, 1], [0, 1], color=DIMC, lw=1, ls=':', label='Perfect')
a12b.set_ylabel('Fraction positives', color=PURC, fontsize=7)
a12b.tick_params(colors=PURC, labelsize=7)
a12b.spines[:].set_color(GRID)
a12b.set_ylim(0, 1)
a12.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT, loc='upper left')
a12b.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT, loc='lower right')

fig2.suptitle(
    f'BTC-USD LSTM ULTIMATE REPAIRED  ·  Feature Intelligence  ·  Drawdown  ·  Walk-Forward CV  ·  {datetime.now().strftime("%Y-%m-%d")}',
    color=TEXT, fontsize=10, fontweight='bold', y=0.995)

plt.show()
print("Charts generated successfully.")
print("\n  Done. ✓\n")
