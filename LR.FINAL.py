import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
import warnings
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TICKER = "BTC-USD"
DATA_START = "2018-01-01"
DATA_END = "2026-05-11"

TEST_DAYS = 180
TRAIN_WINDOW = 600
RETRAIN_EVERY = 10

LR = 0.01
ITERS = 5000
L2 = 0.01

# Risk Management Config
TC = 0.0002               # 0.04% round trip (applied on turnover)
KELLY_FRAC = 0.25         # Conservative Kelly fraction
RR_RATIO = 2.0            # 2:1 Reward to Risk
SL_ATR_MULT = 1.5         # Stop-loss distance in ATRs
MAX_SIZE = 1.0            # Max position leverage

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*78)
print(
    f"  BTC-USD LOGISTIC REGRESSION |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═"*78)
print("  Downloading BTC data...")
raw = yf.download(TICKER, start=DATA_START, end=DATA_END, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def zscore(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)


df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
df['Mom3'] = df['LogRet'].rolling(3).mean()
df['Mom10'] = df['LogRet'].rolling(10).mean()
df['Mom21'] = df['LogRet'].rolling(21).mean()
df['RSI14'] = (rsi(df['Close']) - 50) / 50

sma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BBPos'] = (df['Close'] - sma20) / (2 * std20 + 1e-9)

ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9).mean()
df['MACD'] = macd - signal

df['VolZ'] = zscore(df['Volume'])
rv20 = df['LogRet'].rolling(20).std()
df['VolReg'] = rv20 / (rv20.rolling(60).mean() + 1e-9)
df['HLZ'] = zscore((df['High'] - df['Low']) / df['Close'])

df['EMA9'] = df['Close'].ewm(span=9).mean()
df['EMA21'] = df['Close'].ewm(span=21).mean()
df['EMA_DIFF'] = (df['EMA9'] - df['EMA21']) / df['Close']

df['ATR'] = (df['High'] - df['Low']).rolling(14).mean() / df['Close']
df['RET2'] = df['LogRet'].rolling(2).sum()
df['RET5'] = df['LogRet'].rolling(5).sum()

df['Target'] = (df['LogRet'].shift(-1) > 0).astype(int)

FEATURES = ['LogRet', 'Mom3', 'Mom10', 'Mom21', 'RSI14', 'BBPos', 'MACD',
            'VolZ', 'VolReg', 'HLZ', 'EMA_DIFF', 'ATR', 'RET2', 'RET5']

for feature in FEATURES:
    df[f'L_{feature}'] = df[feature].shift(1)

df.dropna(inplace=True)
LAG_FEATURES = [f'L_{f}' for f in FEATURES]

print(f"  Dataset size: {len(df)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════


class LogisticRegression:
    def __init__(self, lr=0.01, iters=5000, l2=0.01):
        self.lr = lr
        self.iters = iters
        self.l2 = l2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.random.randn(n) * 0.01
        self.b = 0
        for _ in range(self.iters):
            z = X @ self.w + self.b
            p = self.sigmoid(z)
            dw = (1/m) * (X.T @ (p - y)) + (self.l2/m) * self.w
            db = (1/m) * np.sum(p - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def prob(self, X):
        return self.sigmoid(X @ self.w + self.b)

# ══════════════════════════════════════════════════════════════════════════════
# RISK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def kelly_size(prob, direction, rr=RR_RATIO, kf=KELLY_FRAC, cap=MAX_SIZE):
    if direction == 0:
        return 0.0
    p = float(np.clip(prob if direction == 1 else 1.0-prob, 0.01, 0.99))
    f = max(0.0, (p * rr - (1 - p)) / rr)
    return min(kf * f, cap)


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
    trough = int(np.argmin(dd))
    recovered = np.where(wealth[trough:] >= peak[trough])[0]
    recovery = int(recovered[0]) if len(recovered) else len(rets) - trough
    ann_ret = float((wealth[-1] ** (365 / max(len(rets), 1))) - 1)
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
    return dict(max_dd=max_dd, avg_dd=avg_dd, max_dur=max_dur, recovery=recovery, calmar=calmar, wealth=wealth, dd_series=dd)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD TESTING (FINANCIAL ENGINE)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Running walk-forward backtest for {TEST_DAYS} days...")

test_start = len(df) - TEST_DAYS
records = []
cache = {'model': None, 'mu': None, 'sigma': None, 'last_train': -999}

# Financial arrays
sizes, ksizes, bp = np.zeros(TEST_DAYS), np.zeros(
    TEST_DAYS), np.zeros(TEST_DAYS)
sr_gross, sr_daily = np.zeros(TEST_DAYS), np.zeros(TEST_DAYS)
sl_arr, tp_arr = np.full(TEST_DAYS, np.nan), np.full(TEST_DAYS, np.nan)
labels_bt = ["CASH  "] * TEST_DAYS
total_costs_paid = 0.0

for i in range(TEST_DAYS):
    absolute_index = test_start + i

    if (i - cache['last_train']) >= RETRAIN_EVERY or cache['model'] is None:
        train = df.iloc[max(0, absolute_index - TRAIN_WINDOW): absolute_index]
        X_train, y_train = train[LAG_FEATURES].values, train['Target'].values
        mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
        cache['model'] = LogisticRegression(lr=LR, iters=ITERS, l2=L2)
        cache['model'].fit((X_train - mu) / sigma, y_train)
        cache['mu'], cache['sigma'], cache['last_train'] = mu, sigma, i

    row = df.iloc[absolute_index]
    x_scaled = (row[LAG_FEATURES].values - cache['mu']) / cache['sigma']
    prob = cache['model'].prob(x_scaled.reshape(1, -1))[0]
    bp[i] = prob

    # Signal Logic
    direction = 1 if prob > 0.54 else (-1 if prob < 0.46 else 0)
    k_sz = kelly_size(prob, direction)
    if k_sz == 0.0:
        direction = 0

    sizes[i] = direction * k_sz
    ksizes[i] = k_sz
    labels_bt[i] = "LONG  " if direction == 1 else (
        "SHORT " if direction == -1 else "CASH  ")

    # Transaction Cost (Turnover only)
    turnover = abs(sizes[i]) if i == 0 else abs(sizes[i] - sizes[i-1])
    tc_cost = turnover * TC
    total_costs_paid += tc_cost

    # Intraday Execution Logic
    epx, hpx, lpx, cpx = row['Open'], row['High'], row['Low'], row['Close']
    atr = row['L_ATR'] * cpx  # De-normalize ATR
    trade_ret = 0.0

    if direction == 1:
        sl_px, tp_px = epx - atr*SL_ATR_MULT, epx + atr*SL_ATR_MULT*RR_RATIO
        sl_arr[i], tp_arr[i] = sl_px, tp_px
        if lpx <= sl_px:
            trade_ret = (sl_px - epx) / epx
        elif hpx >= tp_px:
            trade_ret = (tp_px - epx) / epx
        else:
            trade_ret = (cpx - epx) / epx

    elif direction == -1:
        sl_px, tp_px = epx + atr*SL_ATR_MULT, epx - atr*SL_ATR_MULT*RR_RATIO
        sl_arr[i], tp_arr[i] = sl_px, tp_px
        if hpx >= sl_px:
            trade_ret = (epx - sl_px) / epx
        elif lpx <= tp_px:
            trade_ret = (epx - tp_px) / epx
        else:
            trade_ret = (epx - cpx) / epx

    gross = trade_ret * k_sz
    sr_gross[i] = gross
    sr_daily[i] = gross - tc_cost

    records.append({
        'Date': df.index[absolute_index], 'Prob': prob,
        'Prediction': 1 if prob > 0.5 else 0, 'Actual': row['Target']
    })

rdf = pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════
cum_strat = (1 + sr_daily).cumprod() - 1
cum_gross = (1 + sr_gross).cumprod() - 1
bh_ret = np.exp(
    df['LogRet'].iloc[test_start: test_start + TEST_DAYS].values) - 1
cum_bh = (1 + bh_ret).cumprod() - 1

sh_net, sh_net_lo, sh_net_hi = sharpe_with_ci(sr_daily)
sh_gross, _, _ = sharpe_with_ci(sr_gross)
sh_bh, sh_bh_lo, sh_bh_hi = sharpe_with_ci(bh_ret)
dd = drawdown_stats(sr_daily)
dd_bh = drawdown_stats(bh_ret)
active_rets = [sr_daily[i] for i in range(TEST_DAYS) if sizes[i] != 0]

wins, losses, active = sum(r > 0 for r in active_rets), sum(
    r < 0 for r in active_rets), len(active_rets)
avg_win = np.mean([r for r in active_rets if r > 0]) if wins else 0.0
avg_loss = np.mean([r for r in active_rets if r < 0]) if losses else 0.0
pf = sum(r for r in active_rets if r > 0) / \
    max(abs(sum(r for r in active_rets if r < 0)), 1e-9)

y_true = rdf['Actual']
y_pred = rdf['Prediction']
y_prob = rdf['Prob']

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc_overall = roc_auc_score(y_true, y_prob)

# ANSI Colors
GRN = "\033[92m"
RED = "\033[91m"
BLU = "\033[94m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"
RST = "\033[0m"
BLD = "\033[1m"
DIM = "\033[2m"

# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL UI REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*105)
print(f"{BLD}  MODEL PERFORMANCE REPORT  —  BTC-USD LOGISTIC REGRESSION{RST}")
print("═"*105)

print(f"\n  {'METRIC':<20} {'TEST SCORE':>11}")
print(f"  {'─'*20} {'─'*11}")
for key, val, lbl in [('acc', acc, 'Accuracy'), ('prec', prec, 'Precision'), ('rec', rec, 'Recall'), ('f1', f1, 'F1-Score'), ('auc', auc_overall, 'ROC-AUC')]:
    v_pct = val * 100
    col_c = GRN if v_pct >= 55 else (YEL if v_pct >= 50 else RED)
    print(f"  {lbl:<20} {col_c}{v_pct:>10.1f}%{RST}")

print(f"\n  Confusion Matrix (TEST):")
cm = confusion_matrix(y_true, y_pred)
print(f"  {'':14} Pred:0   Pred:1")
print(f"  True:0        {cm[0, 0]:>5}    {cm[0, 1]:>5}")
print(f"  True:1        {cm[1, 0]:>5}    {cm[1, 1]:>5}")

print(f"\n  Probability Distribution (TEST, {len(y_prob)} samples):")
bins_conf = [(0, .46, '<46% Short Zone'), (.46, .50, '46–50% Weak Short'),
             (.50, .54, '50–54% Weak Long'), (.54, 1., '>54% Long Zone')]
for lo_b, hi_b, lbl in bins_conf:
    cnt = ((y_prob >= lo_b) & (y_prob < hi_b)).sum()
    bar = "█" * int(cnt/len(y_prob)*40)
    print(f"  {lbl:<22} {cnt:>4}  {bar}")

print("\n" + "═"*105)
print(f"{BLD}  180-DAY BACKTEST  |  Kelly Sizing · TC={TC*100:.2f}% (Turnover) · TP={SL_ATR_MULT*RR_RATIO:.1f}x ATR · SL={SL_ATR_MULT:.1f}x ATR{RST}")
print("═"*105)

hdr = (f"{'DATE':<12} {'PRICE':>9} {'PROB%':>6} {'SIG':<7} {'DIR':>4} {'KELLY':>6} {'SL$':>9} {'TP$':>9} {'GROSS%':>7} {'NET%':>7} {'CUM%':>8}  ST")
print(f"\n  {hdr}")
print("  " + "─"*110)

for i in range(TEST_DAYS):
    s = sizes[i]
    conf = bp[i]
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
    print(f"  {str(rdf['Date'].iloc[i].date()):<12} ${df['Close'].iloc[test_start+i]:>8,.0f} {conf*100:>5.1f}% {col}{labels_bt[i]}{RST} {s:>+4.2f} {ksizes[i]:>5.3f}x {DIM}{sl_s}{RST} {DIM}{tp_s}{RST} {gross:>6.2f}% {nc}{net:>6.2f}%{RST} {cc}{cum:>7.2f}%{RST}  {stat}")

print("  " + "─"*110)


def pline(label, val, fmt='%', col=None, extra=''):
    if col is None:
        col = GRN if val >= 0 else RED
    v_s = f"{val*100:>+8.2f}%" if fmt == '%' else f"{val:>+8.2f}"
    print(f"  {label:<38} {col}{v_s}{RST}{extra}")


print()
pline("Strategy Return (net of TC)", cum_strat[-1])
pline("Strategy Return (gross)", cum_gross[-1])
pline("Transaction Costs Paid Total", -total_costs_paid, col=RED)
pline("Buy & Hold Return", cum_bh[-1])
pline("Alpha (net vs B&H)", cum_strat[-1]-cum_bh[-1])

print()
print(
    f"  {'Sharpe Ratio (net, ann.)':<38} {(GRN if sh_net > 0 else RED)}{sh_net:>+8.2f}{RST}  (95% CI: [{sh_net_lo:+.2f}, {sh_net_hi:+.2f}])")
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
print(f"  {'Profit Factor':<38} {(GRN if pf >= 1 else RED)}{pf:>8.2f}x{RST}")
print(f"  {'Active Days / Total Days':<38}  {active:>4}/{TEST_DAYS}")

# Tomorrow's Prediction
row = df.iloc[-1]
x_scaled = (row[LAG_FEATURES].values - cache['mu']) / cache['sigma']
tm_prob = cache['model'].prob(x_scaled.reshape(1, -1))[0]
tm_dir = 1 if tm_prob > 0.54 else (-1 if tm_prob < 0.46 else 0)
tm_k_sz = kelly_size(tm_prob, tm_dir)
if tm_k_sz == 0.0:
    tm_dir = 0
tm_sz = tm_dir * tm_k_sz

cpx, atr = row['Close'], row['ATR'] * row['Close']
tm_sig = "LONG  " if tm_dir == 1 else ("SHORT " if tm_dir == -1 else "CASH  ")
col_tm = GRN if tm_sz > 0 else (RED if tm_sz < 0 else BLU)
tm_ex = f"  |  SL: ${cpx - atr*SL_ATR_MULT * tm_dir:,.0f}  TP: ${cpx + atr*SL_ATR_MULT*RR_RATIO * tm_dir:,.0f}" if tm_sz != 0 else ""

print("\n" + "═"*105)
print(f"  {BLD}TOMORROW'S SIGNAL:{RST}  {col_tm}{BLD}{tm_sig.strip()}{RST}  | Prob: {tm_prob*100:.1f}%  Size: {tm_sz:+.2f}x  Kelly: {tm_k_sz:.3f}x{tm_ex}")
print(f"  Long Thr: 54.0%  |  Short Thr: 46.0%")
print("═"*105)

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
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
ax_style(a1, f'① ROC Curve  (Test AUC = {auc_overall:.3f})')
fpr_t, tpr_t, _ = roc_curve(y_true, y_prob)
a1.plot(fpr_t, tpr_t, color=GRNC, lw=2,
        label=f'LR Model (AUC={auc_overall:.3f})')
a1.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--', label='Random')
a1.fill_between(fpr_t, tpr_t, alpha=0.15, color=GRNC)
a1.set_xlabel('FPR')
a1.set_ylabel('TPR')
a1.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a2 = fig1.add_subplot(gs1[0, 1])
ax_style(a2, '② Precision–Recall Curve')
pr_p, re_p, _ = precision_recall_curve(y_true, y_prob)
a2.plot(re_p, pr_p, color=BLUC, lw=2)
a2.axhline(0.5, color=GRID, lw=1, ls='--', label='Random')
a2.axvline(0.54, color=YELC, lw=1, ls=':', alpha=0.7, label='Long THR=0.54')
a2.fill_between(re_p, pr_p, alpha=0.10, color=BLUC)
a2.set_xlabel('Recall')
a2.set_ylabel('Precision')
a2.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a3 = fig1.add_subplot(gs1[0, 2])
ax_style(a3, '③ Probability Distribution (Test)')
bns = np.linspace(0, 1, 30)
a3.hist(y_prob[y_true == 0], bins=bns, alpha=0.6, color=REDC, label='Down/SL')
a3.hist(y_prob[y_true == 1], bins=bns, alpha=0.6, color=GRNC, label='Up/TP')
a3.axvline(0.54, color=GRNC, lw=2, ls='--', label='Long THR')
a3.axvline(0.46, color=REDC, lw=2, ls='--', label='Short THR')
a3.set_xlabel('P(Direction)')
a3.set_ylabel('Count')
a3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a4 = fig1.add_subplot(gs1[1, :])
ax_style(a4, f'④ Strategy (net/gross) vs Buy & Hold — {TEST_DAYS} d')
dx = range(TEST_DAYS)
a4.plot(dx, cum_strat*100, color=GRNC, lw=2,
        label='Net (Kelly+TC)', marker='o', ms=2)
a4.plot(dx, cum_gross*100, color=TEAL, lw=1.2, label='Gross (Kelly)', ls=':')
a4.plot(dx, cum_bh*100, color=BLUC, lw=2, label='Buy & Hold', ls='--')
a4.axhline(0, color=GRID, lw=1)
a4.fill_between(dx, cum_strat*100, 0, alpha=0.12, color=GRNC)
a4.set_xlabel('Day')
a4.set_ylabel('Return %')
a4.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

a5 = fig1.add_subplot(gs1[2, :])
ax_style(a5, f'⑤ Daily Net Returns, Signals & Probability — {TEST_DAYS} d')
bc5 = [GRNC if s > 0 else (REDC if s < 0 else GRID) for s in sizes]
a5.bar(dx, sr_daily*100, color=bc5, alpha=0.75)
a5.axhline(0, color=GRID, lw=1)
a5b = a5.twinx()
a5b.plot(dx, bp*100, color=YELC, lw=1.5, alpha=0.8, label='Prob %')
a5b.axhline(54, color=GRNC, lw=0.8, ls='--', alpha=0.4)
a5b.axhline(46, color=REDC, lw=0.8, ls='--', alpha=0.4)
a5b.set_ylabel('Probability %', color=YELC, fontsize=7)
a5b.tick_params(colors=YELC, labelsize=7)
a5b.spines[:].set_color(GRID)
a5b.set_ylim(0, 100)
a5.set_xlabel('Day')
a5.set_ylabel('Net Return %')
pats = [Patch(color=GRNC, label='Long'), Patch(
    color=REDC, label='Short'), Patch(color=GRID, label='Cash')]
a5.legend(handles=pats, fontsize=7, facecolor=PANEL, labelcolor=TEXT)

fig1.suptitle(
    f'BTC-USD LOGISTIC REGRESSION  ·  PERFORMANCE & FINANCIALS  ·  {datetime.now().strftime("%Y-%m-%d")}\n'
    f'Test AUC: {auc_overall:.3f}  ·  Sharpe: {sh_net:.2f} [{sh_net_lo:.2f},{sh_net_hi:.2f}]  ·  PF: {pf:.2f}',
    color=TEXT, fontsize=10, fontweight='bold', y=0.97)

plt.show()

# ─── FIGURE 2: FEATURE INTELLIGENCE & DEEP RISK ─────────────────────────────
fig2 = plt.figure(figsize=(20, 13), facecolor=DARK)
gs2 = gridspec.GridSpec(2, 1, figure=fig2, hspace=0.35)

a6 = fig2.add_subplot(gs2[0, 0])
ax_style(a6, '⑥ Model Features (Final Weights)', 8.5)
final_w = cache['model'].w
sort_idx = np.argsort(np.abs(final_w))[::-1]
sc_w = final_w[sort_idx]
sl_w = [FEATURES[i] for i in sort_idx]
bc6 = [GRNC if v > 0 else REDC for v in sc_w]
a6.barh(range(len(sort_idx)), sc_w, color=bc6, alpha=0.85)
a6.set_yticks(range(len(sort_idx)))
a6.set_yticklabels(sl_w, fontsize=7.5, color=TEXT)
a6.invert_yaxis()
a6.axvline(0, color=GRID, lw=1)
a6.set_xlabel("Weight Magnitude")

a7 = fig2.add_subplot(gs2[1, 0])
ax_style(
    a7, f'⑦ Drawdown Analysis  ·  MaxDD {dd["max_dd"]*100:.1f}%  ·  Calmar {dd["calmar"]:.2f}', 8.5)
wealth_s = (dd['wealth']-1)*100
dd_s = dd['dd_series']*100
a7.plot(dx, wealth_s, color=GRNC, lw=1.8, label='Strategy wealth')
a7.fill_between(dx, dd_s, 0, alpha=0.45, color=REDC, label='Drawdown')
a7.axhline(0, color=GRID, lw=1)
a7b = a7.twinx()
a7b.plot(dx, (dd_bh['wealth']-1)*100, color=BLUC,
         lw=1.5, ls='--', alpha=0.7, label='B&H wealth')
a7b.set_ylabel('B&H Return %', color=BLUC, fontsize=7)
a7b.tick_params(colors=BLUC, labelsize=7)
a7b.spines[:].set_color(GRID)
a7.set_xlabel('Day')
a7.set_ylabel('Strategy Return / Drawdown %')
mdd_i = int(np.argmin(dd['dd_series']))
a7.annotate(f'MaxDD\n{dd["max_dd"]*100:.1f}%', xy=(mdd_i, dd_s[mdd_i]), xytext=(min(mdd_i+5, TEST_DAYS-1),
            dd_s[mdd_i]-2.5), color=REDC, fontsize=7.5, fontweight='bold', arrowprops=dict(arrowstyle='->', color=REDC, lw=1.2))
a7.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='upper left')
a7b.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT, loc='lower right')

fig2.suptitle(
    f'BTC-USD LOGISTIC REGRESSION  ·  FEATURE INTELLIGENCE & RISK  ·  {datetime.now().strftime("%Y-%m-%d")}',
    color=TEXT, fontsize=10, fontweight='bold', y=0.96)

plt.show()
print("  Done. ✓\n")
