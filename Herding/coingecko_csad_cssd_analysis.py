from __future__ import annotations
from matplotlib import cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "crypto_panel_top30_daily_cleaned.csv"

# Use "symbol" if you want columns like BTC, ETH, XRP.
# Use "coin_id" if you want the safest unique names.
WIDE_COLUMN_STYLE = "symbol"   # "symbol" or "coin_id"

MIN_COINS_PER_DAY = 10         # keep only days with at least this many coin returns
FPS = 10
FRAME_STEP = 2

DOT_SIZE = 28
CURRENT_DAY_DOT_SIZE = 90

DARK_GRAY = "#666666"
MED_GRAY  = "#8A8A8A"
LINE_GRAY = "#4F4F4F"

# Output files
OUT_WIDE_PRICES = "crypto_prices_wide.csv"
OUT_WIDE_RETURNS = "crypto_log_returns_wide.csv"
OUT_DAILY_STATS = "daily_csad_cssd.csv"

OUT_CSAD_PNG = "csad_vs_market_return.png"
OUT_CSSD_PNG = "cssd_vs_market_return.png"
OUT_COMPARISON_PNG = "csad_cssd_comparison.png"

OUT_ANIM_MP4 = "csad_cssd_comparison_animation.mp4"
#OUT_ANIM_GIF = "FINAL_csad_cssd_transparent_keynote.gif"
OUT_ANIM_GIF = "UPDATED_csad_cssd_side_by_side_animation.gif"



OUT_PROF_STYLE_HTML = "csad_cssd_prof_style_animation.html"
OUT_PROF_STYLE_CSAD_HTML = "csad_prof_style_animation.html"
OUT_PROF_STYLE_CSSD_HTML = "cssd_prof_style_animation.html"
OUT_LOGRET_CSAD_FIG = "log_returns_and_csad_our_dataset.png"
OUT_LOGRET_CSAD_CSV = "log_returns_and_csad_our_dataset.csv"

OUT_GAMMA2_CONCEPT_GIF = "csad_gamma2_curve_morph_transparent.gif"
OUT_GAMMA2_CONCEPT_FRAME_DIR = "csad_gamma2_curve_morph_frames"




OUT_ANIM_MOV = "csad_cssd_comparison_animation_transparent.mov"
OUT_FRAME_DIR = "csad_cssd_transparent_frames"




##
OUT_CSAD_SCATTER_TS_MOV = "csad_scatter_timeseries_animation_transparent.mov"
OUT_CSSD_SCATTER_TS_MOV = "cssd_scatter_timeseries_animation_transparent.mov"

OUT_CSAD_SCATTER_TS_FRAME_DIR = "csad_scatter_timeseries_frames"
OUT_CSSD_SCATTER_TS_FRAME_DIR = "cssd_scatter_timeseries_frames"

OUT_CSAD_SCATTER_TS_GIF = "csad_scatter_timeseries_animation.gif"
OUT_CSSD_SCATTER_TS_GIF = "cssd_scatter_timeseries_animation.gif"



OUT_CSAD_STACKED_GIF = "csad_scatter_top_timeseries_bottom_transparent.gif"
OUT_CSSD_STACKED_GIF = "cssd_scatter_top_timeseries_bottom_transparent.gif"

OUT_CSAD_STACKED_FRAME_DIR = "csad_stacked_transparent_frames"
OUT_CSSD_STACKED_FRAME_DIR = "cssd_stacked_transparent_frames"


####++

OUT_REG_COMPONENTS_PNG = "csad_cssd_regression_components_over_time.png"
OUT_REG_COMPONENTS_GIF = "csad_cssd_regression_components_animation.gif"
OUT_REG_COMPONENTS_FRAME_DIR = "csad_cssd_regression_components_frames"
####++


###
OUT_TOP5_LOGRET_CSAD_GIF = "top5_log_returns_csad_animation.gif"
OUT_TOP5_LOGRET_CSAD_FRAME_DIR = "top5_log_returns_csad_frames"




TOP_PANEL_COINS = ["BTC", "ETH", "XRP", "BCH", "XMR", "DASH", "ETC", "ZEC", "DOGE", "ADA", "SOL"]

EVENT_LABELS = {
    "2025-07-10": "Bitcoin hits record high",
    "2025-08-08": "SEC ends Ripple case",
    "2026-02-05": "Crypto rout / $2T wiped out",
}

LIGHT_BORDER = "#9A9A9A"
LIGHT_TEXT   = "#7A7A7A"

LIGHTER_GRAY = "#B3B3B3"
CSSD_LIGHT_NAVY = "#5B7DB1"


STATIC_CSAD_LABEL_POS = {
    "Bitcoin hits record high": {"offset": (34, 16), "ha": "left"},
    "SEC ends Ripple case":     {"offset": (18, 34), "ha": "left"},
    "Crypto rout / $2T wiped out": {"offset": (-52, 6), "ha": "left"},
}

STATIC_CSSD_LABEL_POS = {
    "Bitcoin hits record high": {"offset": (46, 18), "ha": "left"},
    "SEC ends Ripple case":     {"offset": (-26, 34), "ha": "right"},
    "Crypto rout / $2T wiped out": {"offset": (-54, 4), "ha": "left"},
}




SCATTER_LABEL_POS = {
    "Bitcoin hits record high": {
        "offset": (38, 30),
        "ha": "left",
        "va": "bottom",
    },
    "SEC ends Ripple case": {
        "offset": (-6, 52),
        "ha": "left",
        "va": "bottom",
    },
    "Crypto rout / $2T wiped out": {
        "offset": (-60, 16),
        "ha": "left",
        "va": "bottom",
    },
}

TS_LABEL_POS = {
    "Bitcoin hits record high": {
        "offset": (-22, 22),
        "ha": "center",
        "va": "bottom",
    },
    "SEC ends Ripple case": {
        "offset": (34, 22),
        "ha": "center",
        "va": "bottom",
    },
    "Crypto rout / $2T wiped out": {
        "offset": (0, 18),
        "ha": "center",
        "va": "bottom",
    },
}



CSSD_SCATTER_LABEL_POS = {
    "Bitcoin hits record high": {
        "offset": (70, 72),
        "ha": "left",
        "va": "bottom",
    },
    "SEC ends Ripple case": {
        "offset": (-28, 54),
        "ha": "right",
        "va": "bottom",
    },
    "Crypto rout / $2T wiped out": {
        "offset": (-70, 12),
        "ha": "left",
        "va": "bottom",
    },
}

CSSD_TS_LABEL_POS = {
    "Bitcoin hits record high": {
        "offset": (-26, 26),
        "ha": "center",
        "va": "bottom",
    },
    "SEC ends Ripple case": {
        "offset": (34, 26),
        "ha": "center",
        "va": "bottom",
    },
    "Crypto rout / $2T wiped out": {
        "offset": (10, 18),
        "ha": "center",
        "va": "bottom",
    },
}


# ============================================================
# HELPERS
# ============================================================
def project_dir() -> Path:
    return Path(__file__).resolve().parent


def make_unique_labels(df: pd.DataFrame, style: str = "symbol") -> dict[str, str]:
    """
    Build a coin_id -> label mapping for wide-format columns.
    """
    meta = (
        df[["coin_id", "symbol"]]
        .drop_duplicates()
        .copy()
        .sort_values(["coin_id", "symbol"])
        .reset_index(drop=True)
    )

    if style == "coin_id":
        meta["label"] = (
            meta["coin_id"]
            .astype(str)
            .str.upper()
            .str.replace("-", "_", regex=False)
        )
        return dict(zip(meta["coin_id"], meta["label"]))

    # Default: symbol-based labels, like BTC, ETH, XRP
    meta["base_label"] = meta["symbol"].astype(str).str.upper().str.strip()

    # If duplicate symbols exist, append coin_id to keep columns unique
    dup_mask = meta["base_label"].duplicated(keep=False)
    meta["label"] = np.where(
        dup_mask,
        meta["base_label"] + "__" + meta["coin_id"].str.upper().str.replace("-", "_", regex=False),
        meta["base_label"],
    )

    return dict(zip(meta["coin_id"], meta["label"]))


def load_clean_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required_cols = {
        "timestamp",
        "coin_id",
        "symbol",
        "price",
        "market_cap",
        "total_volume",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.normalize()

    df["coin_id"] = df["coin_id"].astype(str).str.strip().str.lower()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.lower()

    for col in ["price", "market_cap", "total_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "coin_id", "price"]).copy()
    df = df.sort_values(["coin_id", "date", "timestamp"]).reset_index(drop=True)

    return df


def build_wide_price_matrix(df: pd.DataFrame, style: str = "symbol") -> pd.DataFrame:
    label_map = make_unique_labels(df, style=style)
    df = df.copy()
    df["wide_label"] = df["coin_id"].map(label_map)

    # Keep the last observation for each coin on each day
    df_last = (
        df.sort_values(["coin_id", "date", "timestamp"])
        .drop_duplicates(subset=["coin_id", "date"], keep="last")
        .reset_index(drop=True)
    )

    prices_wide = (
        df_last.pivot(index="date", columns="wide_label", values="price")
        .sort_index()
        .sort_index(axis=1)
    )

    prices_wide.index.name = "date"
    return prices_wide


def compute_log_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices_wide) - np.log(prices_wide.shift(1))
    returns.index.name = "date"
    return returns


def compute_daily_dispersion(
    returns_wide: pd.DataFrame,
    min_coins_per_day: int = 10,
) -> pd.DataFrame:
    n_coins = returns_wide.notna().sum(axis=1)

    # Equal-weight market return across available coins each day
    market_return = returns_wide.mean(axis=1, skipna=True)

    deviations = returns_wide.sub(market_return, axis=0)

    csad = deviations.abs().mean(axis=1, skipna=True)
    cssd = deviations.std(axis=1, ddof=1, skipna=True)

    out = pd.DataFrame(
        {
            "date": returns_wide.index,
            "market_return": market_return,
            "CSAD": csad,
            "CSSD": cssd,
            "n_coins": n_coins,
        }
    )

    out = out[out["n_coins"] >= min_coins_per_day].copy()
    out = out.dropna(subset=["market_return", "CSAD", "CSSD"]).reset_index(drop=True)

    return out


def save_outputs(
    base: Path,
    prices_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    daily_stats: pd.DataFrame,
) -> None:
    prices_wide.to_csv(base / OUT_WIDE_PRICES)
    returns_wide.to_csv(base / OUT_WIDE_RETURNS)
    daily_stats.to_csv(base / OUT_DAILY_STATS, index=False)



####
def build_time_gradient_colors(daily_stats: pd.DataFrame) -> pd.DataFrame:
    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    cmap = plt.colormaps["turbo"].resampled(n)

    mpl_colors = []
    plotly_colors = []

    for i in range(n):
        r, g, b, a = cmap(i)
        mpl_colors.append((r, g, b, a))
        plotly_colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.3f})")

    df["point_color_mpl"] = mpl_colors
    df["point_color_plotly"] = plotly_colors
    return df
####


def make_static_plots(daily_stats: pd.DataFrame, base: Path) -> None:
    df = build_time_gradient_colors(daily_stats)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(df["market_return"], df["CSAD"], s=DOT_SIZE, c=df["point_color_mpl"].tolist())
    ax1.set_xlabel("Market return")
    ax1.set_ylabel("CSAD")
    ax1.set_title("CSAD vs Market Return")
    ax1.grid(alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(base / OUT_CSAD_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(df["market_return"], df["CSSD"], s=DOT_SIZE, c=df["point_color_mpl"].tolist())
    ax2.set_xlabel("Market return")
    ax2.set_ylabel("CSSD")
    ax2.set_title("CSSD vs Market Return")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(base / OUT_CSSD_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------
    # common limits for SAME SCALE
    # -------------------------
    x_min = df["market_return"].min()
    x_max = df["market_return"].max()
    x_pad = max((x_max - x_min) * 0.08, 1e-6)

    # same y-scale for both plots
    y_max_common = max(df["CSAD"].max(), df["CSSD"].max())
    y_pad = max(y_max_common * 0.08, 1e-6)

    # -------------------------
    # left: CSAD
    # -------------------------
    ax3.scatter(
        df["market_return"],
        df["CSAD"],
        s=DOT_SIZE,
        c=df["point_color_mpl"].tolist()
    )
    ax3.set_xlabel("Market return", color=LIGHT_TEXT)
    ax3.set_ylabel("CSAD", color=LIGHT_TEXT)
    ax3.set_title("CSAD vs Market Return", color="black")

    # -------------------------
    # right: CSSD
    # -------------------------
    ax4.scatter(
        df["market_return"],
        df["CSSD"],
        s=DOT_SIZE,
        c=df["point_color_mpl"].tolist()
    )
    ax4.set_xlabel("Market return", color=LIGHT_TEXT)
    ax4.set_ylabel("CSSD", color=LIGHT_TEXT)
    ax4.set_title("CSSD vs Market Return", color="black")

    # -------------------------
    # same axis scale for both
    # -------------------------
    for ax in [ax3, ax4]:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0, y_max_common + y_pad)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)
            ax.spines[side].set_color(LIGHT_BORDER)

        ax.tick_params(axis="both", colors=LIGHT_TEXT, width=0.8)
        ax.grid(False)

    # -------------------------
    # event stars + labels
    # -------------------------
    snapped_events = snap_event_dates_to_data(df, EVENT_LABELS)
    event_rows = df[df["date"].isin(snapped_events.keys())].copy()

    # CSAD stars / labels
    for _, row in event_rows.iterrows():
        label = snapped_events[row["date"]]

        ax3.scatter(
            row["market_return"],
            row["CSAD"],
            s=180,
            marker="*",
            c="red",
            zorder=5
        )

        cfg = STATIC_CSAD_LABEL_POS.get(label, {"offset": (20, 20), "ha": "left"})
        ax3.annotate(
            label,
            xy=(row["market_return"], row["CSAD"]),
            xytext=cfg["offset"],
            textcoords="offset points",
            ha=cfg["ha"],
            va="bottom",
            fontsize=9,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                lw=1.0,
                color=LIGHT_BORDER,
                shrinkA=0,
                shrinkB=4,
            ),
            bbox=dict(
                boxstyle="round,pad=0.12",
                fc=(1, 1, 1, 0.70),
                ec="none",
            ),
        )

    # CSSD stars / labels
    for _, row in event_rows.iterrows():
        label = snapped_events[row["date"]]

        ax4.scatter(
            row["market_return"],
            row["CSSD"],
            s=180,
            marker="*",
            c="red",
            zorder=5
        )

        cfg = STATIC_CSSD_LABEL_POS.get(label, {"offset": (20, 20), "ha": "left"})
        ax4.annotate(
            label,
            xy=(row["market_return"], row["CSSD"]),
            xytext=cfg["offset"],
            textcoords="offset points",
            ha=cfg["ha"],
            va="bottom",
            fontsize=9,
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                lw=1.0,
                color=LIGHT_BORDER,
                shrinkA=0,
                shrinkB=4,
            ),
            bbox=dict(
                boxstyle="round,pad=0.12",
                fc=(1, 1, 1, 0.70),
                ec="none",
            ),
        )

    fig3.tight_layout()
    fig3.savefig(base / OUT_COMPARISON_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig3)



    plt.close(fig3)


def build_time_gradient_colors(daily_stats: pd.DataFrame) -> pd.DataFrame:
    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    cmap = plt.colormaps["turbo"].resampled(n)

    mpl_colors = []
    for i in range(n):
        r, g, b, a = cmap(i)
        mpl_colors.append((r, g, b, a))

    df["point_color_mpl"] = mpl_colors
    return df


##
def snap_event_dates_to_data(daily_stats: pd.DataFrame, event_labels: dict[str, str]) -> dict[pd.Timestamp, str]:
    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    available = df["date"].sort_values().tolist()

    snapped = {}
    for raw_date, label in event_labels.items():
        target = pd.Timestamp(raw_date).normalize()
        nearest = min(available, key=lambda d: abs(d - target))
        snapped[nearest] = label
    return snapped
##









def save_transparent_gif_from_frames(
    frame_dir: Path,
    gif_path: Path,
    fps: int = 12,
    alpha_threshold: int = 5,
) -> None:
    """
    Save transparent GIF from transparent PNG frames.

    Important:
    GIF transparency is binary, not smooth alpha.
    So it may not be as clean as PNG/MOV transparency, but it should work in Keynote.
    """

    from PIL import Image

    frame_files = sorted(frame_dir.glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    frames = []

    for f in frame_files:
        img = Image.open(f).convert("RGBA")

        # Convert to palette mode with 255 colors.
        # Reserve color index 255 for transparency.
        alpha = img.getchannel("A")
        paletted = img.convert("P", palette=Image.ADAPTIVE, colors=255)

        # Pixels with low alpha become fully transparent.
        transparency_mask = Image.eval(
            alpha,
            lambda a: 255 if a <= alpha_threshold else 0
        )

        paletted.paste(255, mask=transparency_mask)
        paletted.info["transparency"] = 255

        frames.append(paletted)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        transparency=255,
        disposal=2,
        optimize=False,
    )

    print(f"Saved transparent GIF -> {gif_path}")


def make_scatter_top_timeseries_bottom_animation(
    daily_stats: pd.DataFrame,
    base: Path,
    metric: str,
    out_gif_name: str,
    out_frame_dir: str,
    event_labels: dict[str, str] | None = None,
    show_event_labels: bool = True,
) -> None:
    """
    Cleaner layout:

    TOP-LEFT   = square scatter plot
    TOP-RIGHT  = empty spacer (for left alignment)
    BOTTOM     = long time-series plot across full width

    Adds event labels back to the red stars.
    """

    df = build_time_gradient_colors(daily_stats)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    snapped_events = snap_event_dates_to_data(df, event_labels or {})

    x_scatter = df["market_return"].to_numpy()
    y_metric = df[metric].to_numpy()

    dates = df["date"].tolist()
    dates_num = mdates.date2num(dates)
    date_labels = df["date"].dt.strftime("%Y-%m-%d").tolist()

    # ------------------------------------------------------------
    # Axis limits
    # ------------------------------------------------------------
    x_min, x_max = np.nanmin(x_scatter), np.nanmax(x_scatter)
    y_min, y_max = np.nanmin(y_metric), np.nanmax(y_metric)

    x_pad = max((x_max - x_min) * 0.10, 1e-6)
    y_pad = max((y_max - y_min) * 0.12, 1e-6)

    date_min, date_max = min(dates), max(dates)

    # ------------------------------------------------------------
    # Style
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 21,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    })

    # ------------------------------------------------------------
    # Figure layout: manual positions for true left alignment
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Figure layout: manual positions for true left alignment
    # ------------------------------------------------------------
    fig_w, fig_h = 13.0, 10.2  # bigger canvas for better resolution

    fig = plt.figure(
        figsize=(fig_w, fig_h),
        facecolor="none",
        edgecolor="none",
    )

    fig.patch.set_alpha(0)

    # Common left edge for both plots
    left = 0.12

    # Top scatter plot
    scatter_width = 0.36
    scatter_height = scatter_width * (fig_w / fig_h)

    # Lower it a bit so title is not chopped
    scatter_bottom = 0.49

    ax1 = fig.add_axes([
        left,
        scatter_bottom,
        scatter_width,
        scatter_height,
    ])

    # Bottom time-series plot
    ts_bottom = 0.08
    ts_width = 0.82
    ts_height = 0.34

    ax2 = fig.add_axes([
        left,
        ts_bottom,
        ts_width,
        ts_height,
    ])

    for ax in [ax1, ax2]:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.grid(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.9)
            ax.spines[side].set_color(MED_GRAY)

        ax.tick_params(
            axis="both",
            colors=MED_GRAY,
            width=0.8,
            labelsize=12,
        )


    # ------------------------------------------------------------
    # TOP scatter plot
    # ------------------------------------------------------------


    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    ax1.set_xlabel("Market return", labelpad=8, color=MED_GRAY)
    ax1.set_ylabel(metric, labelpad=8, color=MED_GRAY)
    ax1.set_title(
        f"{metric} vs Market Return",
        pad=6,
        fontweight="bold",
        loc="left",
        fontsize=20,
        y=1.01,
        color=DARK_GRAY,
    )

    # ------------------------------------------------------------
    # BOTTOM time-series plot
    # ------------------------------------------------------------
    ax2.set_xlim(date_min, date_max)
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)

    ax2.set_xlabel("")
    ax2.set_ylabel(metric, labelpad=8, color=MED_GRAY)
    ax2.set_title(
        f"{metric} over Time",
        pad=10,
        fontweight="bold",
        fontsize=20,
        color=DARK_GRAY,
    )

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # ------------------------------------------------------------
    # Artists
    # ------------------------------------------------------------
    scatter_past = ax1.scatter([], [], s=DOT_SIZE)

    ts_line, = ax2.plot([], [], linewidth=1.5, color=LINE_GRAY)
    ts_current = ax2.scatter([], [], s=CURRENT_DAY_DOT_SIZE, color="red", zorder=5)

    scatter_events = ax1.scatter([], [], s=170, marker="*", c="red", zorder=6)
    ts_events = ax2.scatter([], [], s=150, marker="*", c="red", zorder=6)

    scatter_event_annotations = []
    ts_event_annotations = []

    info_text = fig.text(
        0.77,
        0.945,
        "",
        ha="right",
        va="top",
        fontsize=12,
        color="dimgray",
        fontweight="bold",
    )

    def clear_annotations():
        nonlocal scatter_event_annotations, ts_event_annotations

        for ann in scatter_event_annotations + ts_event_annotations:
            ann.remove()

        scatter_event_annotations = []
        ts_event_annotations = []

    def update(frame: int):
        clear_annotations()

        current_date = df.loc[frame, "date"]

        # --------------------------------------------------------
        # TOP scatter: cumulative points
        # --------------------------------------------------------
        xs = x_scatter[: frame + 1]
        ys = y_metric[: frame + 1]
        colors = df["point_color_mpl"].iloc[: frame + 1].tolist()

        scatter_past.set_offsets(np.column_stack([xs, ys]))
        scatter_past.set_facecolors(colors)
        scatter_past.set_edgecolors(colors)
        scatter_past.set_sizes(np.full(frame + 1, DOT_SIZE))

        # --------------------------------------------------------
        # BOTTOM time series
        # --------------------------------------------------------
        ts_line.set_data(dates, np.where(np.arange(len(dates)) <= frame, y_metric, np.nan))
        ts_current.set_offsets([[dates_num[frame], y_metric[frame]]])

        # --------------------------------------------------------
        # Event markers
        # --------------------------------------------------------
        shown_event_dates = [d for d in snapped_events.keys() if d <= current_date]
        shown_event_rows = df[df["date"].isin(shown_event_dates)].copy()

        if not shown_event_rows.empty:
            scatter_events.set_offsets(
                np.column_stack([
                    shown_event_rows["market_return"],
                    shown_event_rows[metric],
                ])
            )

            ts_events.set_offsets(
                np.column_stack([
                    shown_event_rows["date"].map(mdates.date2num),
                    shown_event_rows[metric],
                ])
            )

            if show_event_labels:
                if metric == "CSAD":
                    scatter_pos_dict = SCATTER_LABEL_POS
                    ts_pos_dict = TS_LABEL_POS
                else:  # CSSD
                    scatter_pos_dict = CSSD_SCATTER_LABEL_POS
                    ts_pos_dict = CSSD_TS_LABEL_POS

                for _, row in shown_event_rows.iterrows():
                    label = snapped_events[row["date"]]

                    s_cfg = scatter_pos_dict.get(label, {
                        "offset": (20, 20),
                        "ha": "left",
                        "va": "bottom",
                    })

                    scatter_event_annotations.append(
                        ax1.annotate(
                            label,
                            xy=(row["market_return"], row[metric]),
                            xytext=s_cfg["offset"],
                            textcoords="offset points",
                            arrowprops=dict(
                                arrowstyle="-",
                                lw=0.9,
                                color=MED_GRAY,
                                shrinkA=0,
                                shrinkB=4,
                            ),
                            fontsize=10,
                            color=MED_GRAY,
                            ha=s_cfg["ha"],
                            va=s_cfg["va"],
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc=(1, 1, 1, 0.72),
                                ec="none",
                            ),
                        )
                    )

                    t_cfg = ts_pos_dict.get(label, {
                        "offset": (0, 16),
                        "ha": "center",
                        "va": "bottom",
                    })

                    ts_event_annotations.append(
                        ax2.annotate(
                            label,
                            xy=(row["date"], row[metric]),
                            xytext=t_cfg["offset"],
                            textcoords="offset points",
                            arrowprops=dict(
                                arrowstyle="-",
                                lw=0.9,
                                color=MED_GRAY,
                                shrinkA=0,
                                shrinkB=4,
                            ),
                            fontsize=10,
                            color=MED_GRAY,
                            ha=t_cfg["ha"],
                            va=t_cfg["va"],
                            bbox=dict(
                                boxstyle="round,pad=0.15",
                                fc=(1, 1, 1, 0.72),
                                ec="none",
                            ),
                        )
                    )
        else:
            scatter_events.set_offsets(np.empty((0, 2)))
            ts_events.set_offsets(np.empty((0, 2)))

        info_text.set_text(
            f"As of {date_labels[frame]}   |   N coins: {int(df.loc[frame, 'n_coins'])}"
        )

        artists = [
            scatter_past,
            ts_line,
            ts_current,
            scatter_events,
            ts_events,
            info_text,
        ]
        artists.extend(scatter_event_annotations)
        artists.extend(ts_event_annotations)

        return artists

    # ------------------------------------------------------------
    # Save transparent PNG frames
    # ------------------------------------------------------------
    frame_dir = base / out_frame_dir

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    frames_to_save = list(range(0, len(df), FRAME_STEP))

    # make sure the final day is included
    if frames_to_save[-1] != len(df) - 1:
        frames_to_save.append(len(df) - 1)

    for out_i, frame in enumerate(frames_to_save):
        update(frame)

        fig.savefig(
            frame_dir / f"frame_{out_i:04d}.png",
            dpi=300,
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
        )
    print(f"Saved transparent {metric} frames -> {frame_dir}")

    # ------------------------------------------------------------
    # Save transparent GIF
    # ------------------------------------------------------------
    gif_path = base / out_gif_name

    save_transparent_gif_from_frames(
        frame_dir=frame_dir,
        gif_path=gif_path,
        fps=FPS,
        alpha_threshold=5,
    )

    plt.close(fig)

def make_animation(
    daily_stats: pd.DataFrame,
    base: Path,
    event_labels: dict[str, str] | None = None,
) -> None:
    """
    Side-by-side animated GIF:

    Left:  CSAD vs Market Return
    Right: CSSD vs Market Return

    Improvements:
    - same x/y scale for both plots
    - lighter gray borders and axis labels
    - non-overlapping event labels
    - transparent GIF output
    """

    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------
    df = build_time_gradient_colors(daily_stats)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    snapped_events = snap_event_dates_to_data(df, event_labels or {})

    x = df["market_return"].to_numpy()
    y_csad = df["CSAD"].to_numpy()
    y_cssd = df["CSSD"].to_numpy()
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

    # ------------------------------------------------------------
    # Shared axis limits: SAME SCALE
    # ------------------------------------------------------------
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x_pad = max((x_max - x_min) * 0.08, 1e-6)

    y_max_common = max(np.nanmax(y_csad), np.nanmax(y_cssd))
    y_pad = max(y_max_common * 0.08, 1e-6)

    # ------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------
    LIGHT_BORDER = "#9A9A9A"
    LIGHT_TEXT = "#7A7A7A"
    LABEL_TEXT = "#5F5F5F"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # ------------------------------------------------------------
    # Event label positions
    # ------------------------------------------------------------
    CSAD_LABEL_POS = {
        "Bitcoin hits record high": {"offset": (38, 28), "ha": "left"},
        "SEC ends Ripple case": {"offset": (-25, 48), "ha": "right"},
        "Crypto rout / $2T wiped out": {"offset": (-58, 8), "ha": "left"},
    }

    CSSD_LABEL_POS = {
        "Bitcoin hits record high": {"offset": (72, 72), "ha": "left"},
        "SEC ends Ripple case": {"offset": (-36, 52), "ha": "right"},
        "Crypto rout / $2T wiped out": {"offset": (-72, 12), "ha": "left"},
    }

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        facecolor="none",
        edgecolor="none",
    )

    fig.patch.set_alpha(0)

    def setup_ax(ax, y_label: str, title: str) -> None:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0, y_max_common + y_pad)

        ax.set_xlabel("Market return", color=LIGHT_TEXT)
        ax.set_ylabel(y_label, color=LIGHT_TEXT)
        ax.set_title(title, color="black", pad=12)

        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.grid(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)
            ax.spines[side].set_color(LIGHT_BORDER)

        ax.tick_params(axis="both", colors=LIGHT_TEXT, width=0.8)

    setup_ax(ax1, "CSAD", "CSAD vs Market Return")
    setup_ax(ax2, "CSSD", "CSSD vs Market Return")

    # ------------------------------------------------------------
    # Artists
    # ------------------------------------------------------------
    scat1_past = ax1.scatter([], [], s=DOT_SIZE)
    scat2_past = ax2.scatter([], [], s=DOT_SIZE)

    scat1_events = ax1.scatter([], [], s=180, marker="*", c="red", zorder=6)
    scat2_events = ax2.scatter([], [], s=180, marker="*", c="red", zorder=6)

    csad_event_annotations = []
    cssd_event_annotations = []

    date_text = fig.text(
        0.5,
        0.02,
        "",
        ha="center",
        va="bottom",
        fontsize=13,
        color="black",
    )

    def clear_annotations():
        nonlocal csad_event_annotations, cssd_event_annotations

        for ann in csad_event_annotations + cssd_event_annotations:
            ann.remove()

        csad_event_annotations = []
        cssd_event_annotations = []

    def update(frame: int):
        clear_annotations()

        # --------------------------------------------------------
        # Cumulative point clouds
        # --------------------------------------------------------
        xs = x[: frame + 1]
        colors = df["point_color_mpl"].iloc[: frame + 1].tolist()

        csad_points = np.column_stack([xs, y_csad[: frame + 1]])
        cssd_points = np.column_stack([xs, y_cssd[: frame + 1]])

        scat1_past.set_offsets(csad_points)
        scat1_past.set_facecolors(colors)
        scat1_past.set_edgecolors(colors)
        scat1_past.set_sizes(np.full(frame + 1, DOT_SIZE))

        scat2_past.set_offsets(cssd_points)
        scat2_past.set_facecolors(colors)
        scat2_past.set_edgecolors(colors)
        scat2_past.set_sizes(np.full(frame + 1, DOT_SIZE))

        # --------------------------------------------------------
        # Events reached so far
        # --------------------------------------------------------
        current_date = df.loc[frame, "date"]
        shown_event_dates = [d for d in snapped_events.keys() if d <= current_date]
        shown_event_rows = df[df["date"].isin(shown_event_dates)].copy()

        if not shown_event_rows.empty:
            scat1_events.set_offsets(
                np.column_stack([
                    shown_event_rows["market_return"],
                    shown_event_rows["CSAD"],
                ])
            )

            scat2_events.set_offsets(
                np.column_stack([
                    shown_event_rows["market_return"],
                    shown_event_rows["CSSD"],
                ])
            )

            for _, row in shown_event_rows.iterrows():
                label = snapped_events[row["date"]]

                # -----------------------------
                # CSAD event label
                # -----------------------------
                cfg1 = CSAD_LABEL_POS.get(label, {
                    "offset": (25, 25),
                    "ha": "left",
                })

                csad_event_annotations.append(
                    ax1.annotate(
                        label,
                        xy=(row["market_return"], row["CSAD"]),
                        xytext=cfg1["offset"],
                        textcoords="offset points",
                        ha=cfg1["ha"],
                        va="bottom",
                        fontsize=9,
                        color=LABEL_TEXT,
                        arrowprops=dict(
                            arrowstyle="-",
                            lw=1.0,
                            color=LIGHT_BORDER,
                            shrinkA=0,
                            shrinkB=4,
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.12",
                            fc=(1, 1, 1, 0.72),
                            ec="none",
                        ),
                    )
                )

                # -----------------------------
                # CSSD event label
                # -----------------------------
                cfg2 = CSSD_LABEL_POS.get(label, {
                    "offset": (25, 25),
                    "ha": "left",
                })

                cssd_event_annotations.append(
                    ax2.annotate(
                        label,
                        xy=(row["market_return"], row["CSSD"]),
                        xytext=cfg2["offset"],
                        textcoords="offset points",
                        ha=cfg2["ha"],
                        va="bottom",
                        fontsize=9,
                        color=LABEL_TEXT,
                        arrowprops=dict(
                            arrowstyle="-",
                            lw=1.0,
                            color=LIGHT_BORDER,
                            shrinkA=0,
                            shrinkB=4,
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.12",
                            fc=(1, 1, 1, 0.72),
                            ec="none",
                        ),
                    )
                )

        else:
            scat1_events.set_offsets(np.empty((0, 2)))
            scat2_events.set_offsets(np.empty((0, 2)))

        date_text.set_text(
            f"Date: {dates[frame]} | N coins: {int(df.loc[frame, 'n_coins'])}"
        )

        artists = [
            scat1_past,
            scat2_past,
            scat1_events,
            scat2_events,
            date_text,
        ]

        artists.extend(csad_event_annotations)
        artists.extend(cssd_event_annotations)

        return artists

    # ------------------------------------------------------------
    # Save transparent frames
    # ------------------------------------------------------------
    frame_dir = base / OUT_FRAME_DIR

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    # Use FRAME_STEP if it exists. Otherwise default to 2.
    frame_step = globals().get("FRAME_STEP", 2)

    frames_to_save = list(range(0, len(df), frame_step))

    if frames_to_save[-1] != len(df) - 1:
        frames_to_save.append(len(df) - 1)

    for out_i, frame in enumerate(frames_to_save):
        update(frame)

        fig.savefig(
            frame_dir / f"frame_{out_i:04d}.png",
            dpi=260,
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
        )

    print(f"Saved transparent frames -> {frame_dir}")

    # ------------------------------------------------------------
    # Save transparent GIF
    # ------------------------------------------------------------
    from PIL import Image

    gif_path = base / OUT_ANIM_GIF
    print(f"GIF will be saved to: {gif_path.resolve()}")

    frame_files = sorted(frame_dir.glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    gif_frames = []

    for f in frame_files:
        img = Image.open(f).convert("RGBA")
        alpha = img.getchannel("A")

        paletted = img.convert("P", palette=Image.ADAPTIVE, colors=255)

        transparency_mask = Image.eval(
            alpha,
            lambda a: 255 if a <= 5 else 0,
        )

        paletted.paste(255, mask=transparency_mask)
        paletted.info["transparency"] = 255

        gif_frames.append(paletted)

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        transparency=255,
        disposal=2,
        optimize=False,
    )

    print(f"Saved updated side-by-side GIF -> {gif_path}")
    print(f"GIF exists: {gif_path.exists()}")
    print(f"GIF size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    plt.close(fig)


def rgba_from_cmap(cmap_name: str, n: int) -> list[str]:
    cmap = cm.get_cmap(cmap_name, n)
    colors = []
    for i in range(n):
        r, g, b, a = cmap(i)
        colors.append(f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.3f})")
    return colors


def make_prof_style_animation(
    daily_stats: pd.DataFrame,
    base: Path,
    highlight_labels: dict[str, str] | None = None,
) -> None:
    ...




def snap_dates_to_index(index_like, event_labels: dict[str, str]) -> dict[pd.Timestamp, str]:
    idx = pd.to_datetime(pd.Index(index_like)).normalize().sort_values()
    snapped = {}
    for raw_date, label in event_labels.items():
        target = pd.Timestamp(raw_date).normalize()
        nearest = min(idx, key=lambda d: abs(d - target))
        snapped[nearest] = label
    return snapped


def make_log_returns_and_csad_figure(
    returns_wide: pd.DataFrame,
    daily_stats: pd.DataFrame,
    base: Path,
    event_labels: dict[str, str] | None = None,
) -> None:
    # ---------- prepare data ----------
    ret = returns_wide.copy()
    ret.index = pd.to_datetime(ret.index)
    ret = ret.sort_index()

    stats = daily_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"])
    stats = stats.sort_values("date").set_index("date")

    # keep common dates only
    common_idx = ret.index.intersection(stats.index)
    ret = ret.loc[common_idx]
    stats = stats.loc[common_idx]

    # choose top-panel series that actually exist
    available_cols = set(ret.columns.tolist())
    chosen = [c for c in TOP_PANEL_COINS if c in available_cols]

    # if some preferred labels do not exist, fill with first available columns
    if len(chosen) < 8:
        extras = [c for c in ret.columns.tolist() if c not in chosen]
        chosen.extend(extras[: max(0, 8 - len(chosen))])

    # add market return proxy as top-panel benchmark
    ret_plot = ret[chosen].copy()
    #ret_plot["MKT"] = stats["market_return"]
    ret_plot["Market Avg."] = stats["market_return"]

    # save combined CSV used for the figure
    combined = ret_plot.copy()
    combined["CSAD"] = stats["CSAD"]
    combined.to_csv(base / OUT_LOGRET_CSAD_CSV)



    ##

    # save combined CSV used for the figure
    combined = ret_plot.copy()
    combined["CSAD"] = stats["CSAD"]

    # Use existing market average if available
    if "Market Avg." in combined.columns:
        combined["market_return"] = combined["Market Avg."]
    else:
        coin_return_cols = [
            col for col in ret_plot.columns
            if col not in ["CSAD", "CSSD", "Market Avg.", "market_return"]
        ]
        combined["market_return"] = ret_plot[coin_return_cols].mean(axis=1, skipna=True)

    combined.to_csv(base / OUT_LOGRET_CSAD_CSV)

    # =========================
    # COINGECKO CSAD HERDING REGRESSION
    # =========================
    import statsmodels.api as sm

    df_reg = combined.copy()

    df_reg["abs_Rm"] = df_reg["market_return"].abs()
    df_reg["sq_Rm"] = df_reg["market_return"] ** 2

    df_reg = df_reg.dropna(subset=["CSAD", "abs_Rm", "sq_Rm"])

    y = df_reg["CSAD"]
    X = df_reg[["abs_Rm", "sq_Rm"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    print(model.summary())

    with open(base / "coingecko_csad_herding_regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    gamma2 = model.params["sq_Rm"]
    p_gamma2 = model.pvalues["sq_Rm"]

    print("\nHerding interpretation:")
    print(f"gamma_2 coefficient: {gamma2:.6f}")
    print(f"p-value: {p_gamma2:.6f}")

    if gamma2 < 0 and p_gamma2 < 0.05:
        print("Evidence of herding: gamma_2 is negative and statistically significant at the 5% level.")
    else:
        print("No strong evidence of herding: gamma_2 is not both negative and statistically significant.")

    ##

    # ---------- figure ----------
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        facecolor=(0, 0, 0, 0)
    )

    fig.patch.set_alpha(0)
    fig.patch.set_facecolor((0, 0, 0, 0))

    for ax in [ax1, ax2]:
        ax.set_facecolor((0, 0, 0, 0))
        ax.patch.set_alpha(0)
        ax.patch.set_visible(False)
        ax.grid(False)

    # top: log returns
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    line_colors = {}

    # Highlight only key assets; show the rest in gray
    HIGHLIGHT_COINS = {
        "BTC": "tab:blue",
        "ETH": "tab:orange",
        "SOL": "tab:green",
        "Market Avg.": "black",
    }

    line_colors = {}

    for col in ret_plot.columns:
        if col in HIGHLIGHT_COINS:
            color = HIGHLIGHT_COINS[col]
            lw = 2.2 if col == "Market Avg." else 1.8
            alpha = 0.95
            zorder = 3
        else:
            color = "lightgray"
            lw = 0.8
            alpha = 0.35
            zorder = 1

        line_colors[col] = color
        ax1.plot(
            ret_plot.index,
            ret_plot[col],
            linewidth=lw,
            alpha=alpha,
            label=col,
            color=color,
            zorder=zorder,
        )

    ax1.set_ylabel("Log return")
    ax1.set_title("Log Returns and CSAD", fontsize=24, color="red", pad=16)

    # bottom: CSAD
    ax2.plot(stats.index, stats["CSAD"], linewidth=1.1, color="black")
    ax2.set_ylabel("CSAD")

    # ---------- better event windows + labels ----------
    event_windows = [
        {
            "start": "2025-07-06",
            "end": "2025-07-14",
            "label": "Bitcoin hits\nrecord high",
            "text_y": 1.05,
        },
        {
            "start": "2025-08-04",
            "end": "2025-08-12",
            "label": "SEC ends\nRipple case",
            "text_y": 1.12,
        },
        {
            "start": "2026-02-01",
            "end": "2026-02-09",
            "label": "Crypto rout /\n$2T wiped out",
            "text_y": 1.05,
        },
    ]

    top_ymin, top_ymax = ax1.get_ylim()
    bottom_ymin, bottom_ymax = ax2.get_ylim()

    for ev in event_windows:
        start = pd.to_datetime(ev["start"])
        end = pd.to_datetime(ev["end"])
        label = ev["label"]
        text_y = ev["text_y"]

        rect_top = Rectangle(
            (mdates.date2num(start), top_ymin),
            mdates.date2num(end) - mdates.date2num(start),
            top_ymax - top_ymin,
            fill=False,
            edgecolor="dodgerblue",
            linewidth=2,
            linestyle=(0, (3, 3)),
        )
        ax1.add_patch(rect_top)

        rect_bottom = Rectangle(
            (mdates.date2num(start), bottom_ymin),
            mdates.date2num(end) - mdates.date2num(start),
            bottom_ymax - bottom_ymin,
            fill=False,
            edgecolor="dodgerblue",
            linewidth=2,
            linestyle=(0, (3, 3)),
        )
        ax2.add_patch(rect_bottom)




    #ax2.set_ylabel("CSAD")

    # ---------- event windows + labels ----------


    # ---------- styling ----------
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")

    ax1.grid(False)
    ax2.grid(False)

    # right-side legend like prof style
    legend_x = 1.02
    legend_y = 0.95
    step = 0.085
    for i, col in enumerate(ret_plot.columns):
        ax1.text(
            legend_x,
            legend_y - i * step,
            col,
            transform=ax1.transAxes,
            color=line_colors[col],
            fontsize=18,
            fontweight="bold",
            va="top",
        )

    # sample period text
    sample_text = f"Sample period: {common_idx.min().date()} - {common_idx.max().date()}"
    fig.text(0.05, 0.88, sample_text, fontsize=16)

    fig.tight_layout(rect=[0, 0, 0.88, 0.88])
    fig.savefig(base / OUT_LOGRET_CSAD_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved log returns + CSAD figure -> {base / OUT_LOGRET_CSAD_FIG}")
    print(f"Saved log returns + CSAD data -> {base / OUT_LOGRET_CSAD_CSV}")








###

def top_n_labels_by_latest_market_cap(
    panel: pd.DataFrame,
    style: str = "symbol",
    n: int = 5,
) -> list[str]:
    """
    Pick top n coins by latest available market cap.
    Returns labels matching the wide return columns, e.g. BTC, ETH.
    """
    tmp = panel.copy()

    if "date" not in tmp.columns:
        tmp["date"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()

    label_map = make_unique_labels(tmp, style=style)
    tmp["wide_label"] = tmp["coin_id"].map(label_map)

    latest_date = tmp["date"].max()

    latest = (
        tmp[tmp["date"] == latest_date]
        .sort_values(["coin_id", "timestamp"])
        .drop_duplicates(subset=["coin_id"], keep="last")
        .copy()
    )

    latest = latest.dropna(subset=["market_cap"])
    latest = latest.sort_values("market_cap", ascending=False)

    return latest["wide_label"].head(n).tolist()


def make_top5_logreturns_csad_gif(
    returns_wide: pd.DataFrame,
    daily_stats: pd.DataFrame,
    base: Path,
    top_coins: list[str],
    out_gif_name: str = OUT_TOP5_LOGRET_CSAD_GIF,
    out_frame_dir: str = OUT_TOP5_LOGRET_CSAD_FRAME_DIR,
    frame_step: int = 4,
    fps: int = 10,
    target_duration_sec: int = 20,
) -> None:
    """
    Transparent GIF for Keynote:
    Top panel: top 5 coin log returns over time
    Bottom panel: CSAD time series
    Text box: current date, coin returns, CSAD
    """

    # -----------------------------
    # Prepare data
    # -----------------------------
    ret = returns_wide.copy()
    ret.index = pd.to_datetime(ret.index).normalize()
    ret = ret.sort_index()

    stats = daily_stats.copy()
    stats["date"] = pd.to_datetime(stats["date"]).dt.normalize()
    stats = stats.sort_values("date").set_index("date")

    common_idx = ret.index.intersection(stats.index)

    ret = ret.loc[common_idx]
    stats = stats.loc[common_idx]

    chosen = [c for c in top_coins if c in ret.columns]

    if len(chosen) < 5:
        extras = [c for c in ret.columns if c not in chosen]
        extras = sorted(extras, key=lambda c: ret[c].notna().sum(), reverse=True)
        chosen.extend(extras[: 5 - len(chosen)])

    chosen = chosen[:5]

    ret_plot = ret[chosen].copy()
    csad = stats["CSAD"].copy()

    print("Top 5 coins used in GIF:", chosen)
    print("GIF date range:", common_idx.min(), "to", common_idx.max())
    print("Frames before step:", len(common_idx))

    # -----------------------------
    # Style for dark Keynote background
    # -----------------------------
    TEXT_COLOR = "#8E8E8E"  # darker gray for text
    AXIS_COLOR = "#8E8E8E"  # darker gray for axis labels/ticks
    CSAD_COLOR = "#3A67A8"  # slightly lighter navy blue

    coin_colors = {
        chosen[0]: "#1f77b4",
        chosen[1]: "#ff7f0e",
        chosen[2]: "#2ca02c",
        chosen[3]: "#d62728",
        chosen[4]: "#9467bd",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 160,
        "savefig.dpi": 220,
    })

    # -----------------------------
    # Figure
    # -----------------------------
    fig = plt.figure(figsize=(13.5, 8.2), facecolor="none")
    fig.patch.set_alpha(0)

    ax1 = fig.add_axes([0.08, 0.56, 0.72, 0.34])
    ax2 = fig.add_axes([0.08, 0.13, 0.72, 0.30])

    for ax in [ax1, ax2]:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.grid(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_color(AXIS_COLOR)
            ax.spines[side].set_linewidth(1.0)

        ax.tick_params(axis="both", colors=AXIS_COLOR)

    # Axis limits
    y1_min = np.nanmin(ret_plot.to_numpy())
    y1_max = np.nanmax(ret_plot.to_numpy())
    y1_pad = max((y1_max - y1_min) * 0.10, 1e-6)

    y2_min = 0
    y2_max = np.nanmax(csad.to_numpy())
    y2_pad = max(y2_max * 0.12, 1e-6)

    ax1.set_xlim(common_idx.min(), common_idx.max())
    ax1.set_ylim(y1_min - y1_pad, y1_max + y1_pad)

    ax2.set_xlim(common_idx.min(), common_idx.max())
    ax2.set_ylim(y2_min, y2_max + y2_pad)

    ax1.set_ylabel("Log return", color=TEXT_COLOR)
    ax2.set_ylabel("CSAD", color=TEXT_COLOR)

    ax1.set_xticklabels([])

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", color=AXIS_COLOR)

    # -----------------------------
    # Artists
    # -----------------------------
    coin_lines = {}
    coin_dots = {}

    for coin in chosen:
        line, = ax1.plot([], [], lw=1.3, color=coin_colors[coin], alpha=0.95, label=coin)
        dot = ax1.scatter([], [], s=55, color=coin_colors[coin], zorder=5)
        coin_lines[coin] = line
        coin_dots[coin] = dot

    csad_line, = ax2.plot([], [], lw=1.8, color=CSAD_COLOR)
    csad_dot = ax2.scatter([], [], s=70, color=CSAD_COLOR, zorder=5)

    vline1 = ax1.axvline(common_idx[0], color="white", lw=0.9, alpha=0.55)
    vline2 = ax2.axvline(common_idx[0], color="white", lw=0.9, alpha=0.55)

    info_text = fig.text(
        0.81,  # move left so it does not get cropped
        0.90,
        "",
        ha="left",
        va="top",
        fontsize=12,
        color=TEXT_COLOR,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc=(1, 1, 1, 0.0),  # fully transparent background
            ec=(1, 1, 1, 0.0),  # fully transparent border
        ),
    )
    # Legend
    legend_y = 0.48
    for i, coin in enumerate(chosen):
        fig.text(
            0.81,  # move legend left also
            legend_y - i * 0.045,
            coin,
            color=coin_colors[coin],
            fontsize=14,
            fontweight="bold",
            ha="left",
        )

    dates = list(common_idx)
    dates_num = mdates.date2num(dates)

    # -----------------------------
    # Frame folder
    # -----------------------------
    frame_dir = base / out_frame_dir

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    target_frames = max(2, int(fps * target_duration_sec))
    auto_frame_step = max(1, int(np.ceil(len(dates) / target_frames)))

    frames_to_save = list(range(0, len(dates), auto_frame_step))
    if frames_to_save[-1] != len(dates) - 1:
        frames_to_save.append(len(dates) - 1)

    actual_duration = len(frames_to_save) / fps
    print(f"Using frame_step={auto_frame_step}")
    print(f"Saved frames={len(frames_to_save)}")
    print(f"Approx. duration={actual_duration:.1f} seconds")

    # -----------------------------
    # Render frames
    # -----------------------------
    for out_i, frame in enumerate(frames_to_save):
        current_dates = dates[: frame + 1]
        current_date = dates[frame]

        for coin in chosen:
            y = ret_plot[coin].iloc[: frame + 1]
            coin_lines[coin].set_data(current_dates, y)

            current_y = ret_plot[coin].iloc[frame]
            if pd.notna(current_y):
                coin_dots[coin].set_offsets([[dates_num[frame], current_y]])
            else:
                coin_dots[coin].set_offsets(np.empty((0, 2)))

        csad_line.set_data(current_dates, csad.iloc[: frame + 1])
        csad_dot.set_offsets([[dates_num[frame], csad.iloc[frame]]])

        vline1.set_xdata([current_date, current_date])
        vline2.set_xdata([current_date, current_date])

        value_lines = [f"Date: {current_date.strftime('%Y-%m-%d')}"]

        for coin in chosen:
            val = ret_plot[coin].iloc[frame]
            if pd.notna(val):
                value_lines.append(f"{coin}: {val * 100:+.2f}%")
            else:
                value_lines.append(f"{coin}: NA")

        value_lines.append(f"CSAD: {csad.iloc[frame]:.4f}")

        info_text.set_text("\n".join(value_lines))

        fig.savefig(
            frame_dir / f"frame_{out_i:04d}.png",
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
            dpi=220,
        )

    print(f"Saved transparent frames -> {frame_dir}")

    # -----------------------------
    # Save transparent GIF
    # -----------------------------
    gif_path = base / out_gif_name

    save_transparent_gif_from_frames(
        frame_dir=frame_dir,
        gif_path=gif_path,
        fps=fps,
        alpha_threshold=5,
    )

    print(f"Saved top-5 log returns + CSAD GIF -> {gif_path}")
    print(f"GIF exists: {gif_path.exists()}")
    print(f"GIF size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    plt.close(fig)

###





##++
def make_regression_component_plot(
    daily_stats: pd.DataFrame,
    base: Path,
    tail_q: float = 0.05,
) -> None:
    """
    Plot the key regression components for CSAD and CSSD over time.

    Top panel:
        CSAD nonlinear component = gamma_2 * R_m,t^2

    Bottom panel:
        CSSD extreme-market dummy component = beta_L * D_L + beta_U * D_U
    """

    import statsmodels.api as sm

    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ============================================================
    # 1. CSAD regression
    # CSAD_t = alpha + gamma1 |Rm,t| + gamma2 Rm,t^2 + error
    # ============================================================
    df["abs_Rm"] = df["market_return"].abs()
    df["sq_Rm"] = df["market_return"] ** 2

    csad_reg = df.dropna(subset=["CSAD", "abs_Rm", "sq_Rm"]).copy()

    y_csad = csad_reg["CSAD"]
    X_csad = csad_reg[["abs_Rm", "sq_Rm"]]
    X_csad = sm.add_constant(X_csad)

    csad_model = sm.OLS(y_csad, X_csad).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
    )

    gamma2 = csad_model.params["sq_Rm"]
    t_gamma2 = csad_model.tvalues["sq_Rm"]
    p_gamma2 = csad_model.pvalues["sq_Rm"]

    csad_reg["csad_nonlinear_component"] = gamma2 * csad_reg["sq_Rm"]

    # ============================================================
    # 2. CSSD regression
    # CSSD_t = alpha + beta_L D_L + beta_U D_U + error
    # ============================================================
    lower_cutoff = df["market_return"].quantile(tail_q)
    upper_cutoff = df["market_return"].quantile(1 - tail_q)

    df["D_L"] = (df["market_return"] <= lower_cutoff).astype(int)
    df["D_U"] = (df["market_return"] >= upper_cutoff).astype(int)

    cssd_reg = df.dropna(subset=["CSSD", "D_L", "D_U"]).copy()

    y_cssd = cssd_reg["CSSD"]
    X_cssd = cssd_reg[["D_L", "D_U"]]
    X_cssd = sm.add_constant(X_cssd)

    cssd_model = sm.OLS(y_cssd, X_cssd).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
    )

    beta_L = cssd_model.params["D_L"]
    beta_U = cssd_model.params["D_U"]

    t_beta_L = cssd_model.tvalues["D_L"]
    t_beta_U = cssd_model.tvalues["D_U"]

    p_beta_L = cssd_model.pvalues["D_L"]
    p_beta_U = cssd_model.pvalues["D_U"]

    cssd_reg["cssd_extreme_component"] = (
        beta_L * cssd_reg["D_L"] + beta_U * cssd_reg["D_U"]
    )

    # ============================================================
    # 3. Plot
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        facecolor="none",
    )

    fig.patch.set_alpha(0)

    for ax in [ax1, ax2]:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.grid(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(LIGHTER_GRAY)
            ax.spines[side].set_linewidth(1.0)

        ax.tick_params(axis="both", colors=LIGHTER_GRAY)
        ax.axhline(0, linewidth=1.0, color=LIGHTER_GRAY, alpha=0.7)

    # -------------------------
    # Top: CSAD nonlinear part
    # -------------------------
    ax1.plot(
        csad_reg["date"],
        csad_reg["csad_nonlinear_component"],
        linewidth=1.8,
        color=LINE_GRAY,
    )

    ax1.set_ylabel(r"$\gamma_2 R_{m,t}^2$", color=MED_GRAY)
    ax1.set_title(
        "CSAD Nonlinear Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
    )

    csad_text = (
        f"CSAD regression\n"
        f"γ₂ = {gamma2:.4f}\n"
        f"t = {t_gamma2:.2f}\n"
        f"p = {p_gamma2:.3f}"
    )

    ax1.text(
        0.00,
        1.03,
        csad_text,
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        clip_on=False,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.75), ec="none"),
    )

    # -------------------------
    # Bottom: CSSD dummy component
    # -------------------------
    ax2.bar(
        cssd_reg["date"],
        cssd_reg["cssd_extreme_component"],
        width=2.0,
        color=CSSD_LIGHT_NAVY,
        alpha=0.85,
    )

    ax2.set_ylabel(r"$\beta_L D_t^L + \beta_U D_t^U$", color=MED_GRAY)
    ax2.set_title(
        "CSSD Extreme-Market Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
    )

    cssd_text = (
        f"CSSD regression, tail = {int(tail_q * 100)}%\n"
        f"βL = {beta_L:.4f}, pL = {p_beta_L:.3f}\n"
        f"βU = {beta_U:.4f}, pU = {p_beta_U:.3f}"
    )

    ax2.text(
        0.00,
        1.03,
        cssd_text,
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        clip_on=False,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.75), ec="none"),
    )

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(
        "Regression-Based Herding Components Over Time",
        fontsize=22,
        fontweight="bold",
        color=DARK_GRAY,
        y=0.98,
    )

    fig.subplots_adjust(top=0.84, bottom=0.10, hspace=0.48)

    out_path = base / OUT_REG_COMPONENTS_PNG
    fig.savefig(out_path, dpi=240, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Saved regression component plot -> {out_path}")

    # Save regression summaries
    with open(base / "csad_cssd_regression_components_summary.txt", "w") as f:
        f.write("CSAD regression:\n")
        f.write(csad_model.summary().as_text())
        f.write("\n\n")
        f.write("CSSD regression:\n")
        f.write(cssd_model.summary().as_text())


def make_regression_component_animation(
    daily_stats: pd.DataFrame,
    base: Path,
    tail_q: float = 0.05,
    fps: int = FPS,
) -> None:
    """
    Transparent GIF animation for Keynote.

    Top panel:
        CSAD nonlinear component = gamma_2 * R_m,t^2

    Bottom panel:
        CSSD extreme-market component = beta_L * D_L + beta_U * D_U
    """

    import statsmodels.api as sm

    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ============================================================
    # 1. CSAD regression
    # ============================================================
    df["abs_Rm"] = df["market_return"].abs()
    df["sq_Rm"] = df["market_return"] ** 2

    csad_reg = df.dropna(subset=["CSAD", "abs_Rm", "sq_Rm"]).copy()

    y_csad = csad_reg["CSAD"]
    X_csad = sm.add_constant(csad_reg[["abs_Rm", "sq_Rm"]])

    csad_model = sm.OLS(y_csad, X_csad).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
    )

    gamma2 = csad_model.params["sq_Rm"]
    t_gamma2 = csad_model.tvalues["sq_Rm"]
    p_gamma2 = csad_model.pvalues["sq_Rm"]

    csad_reg["csad_nonlinear_component"] = gamma2 * csad_reg["sq_Rm"]

    # ============================================================
    # 2. CSSD regression
    # ============================================================
    lower_cutoff = df["market_return"].quantile(tail_q)
    upper_cutoff = df["market_return"].quantile(1 - tail_q)

    df["D_L"] = (df["market_return"] <= lower_cutoff).astype(int)
    df["D_U"] = (df["market_return"] >= upper_cutoff).astype(int)

    cssd_reg = df.dropna(subset=["CSSD", "D_L", "D_U"]).copy()

    y_cssd = cssd_reg["CSSD"]
    X_cssd = sm.add_constant(cssd_reg[["D_L", "D_U"]])

    cssd_model = sm.OLS(y_cssd, X_cssd).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 3},
    )

    beta_L = cssd_model.params["D_L"]
    beta_U = cssd_model.params["D_U"]

    p_beta_L = cssd_model.pvalues["D_L"]
    p_beta_U = cssd_model.pvalues["D_U"]

    cssd_reg["cssd_extreme_component"] = (
        beta_L * cssd_reg["D_L"] + beta_U * cssd_reg["D_U"]
    )

    # ============================================================
    # 3. Prepare animation data
    # ============================================================
    dates = csad_reg["date"].tolist()
    dates_num = mdates.date2num(dates)

    csad_component = csad_reg["csad_nonlinear_component"].to_numpy()
    cssd_component = cssd_reg["cssd_extreme_component"].to_numpy()

    y1_min = min(0, np.nanmin(csad_component))
    y1_max = max(0, np.nanmax(csad_component))
    y1_pad = max((y1_max - y1_min) * 0.15, 1e-6)

    y2_min = min(0, np.nanmin(cssd_component))
    y2_max = max(0, np.nanmax(cssd_component))
    y2_pad = max((y2_max - y2_min) * 0.15, 1e-6)

    # ============================================================
    # 4. Figure setup
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        facecolor="none",
    )

    fig.patch.set_alpha(0)

    for ax in [ax1, ax2]:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0)
        ax.grid(False)

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(MED_GRAY)
            ax.spines[side].set_linewidth(1.0)

        ax.tick_params(axis="both", colors=MED_GRAY)
        ax.axhline(0, linewidth=1.0, color=MED_GRAY, alpha=0.7)

    ax1.set_xlim(min(dates), max(dates))
    ax1.set_ylim(y1_min - y1_pad, y1_max + y1_pad)

    ax2.set_xlim(min(dates), max(dates))
    ax2.set_ylim(y2_min - y2_pad, y2_max + y2_pad)

    ax1.set_title(
        "CSAD Nonlinear Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
    )

    ax2.set_title(
        "CSSD Extreme-Market Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
    )

    ax1.set_ylabel(r"$\gamma_2 R_{m,t}^2$", color=DARK_GRAY)
    ax2.set_ylabel(r"$\beta_L D_t^L + \beta_U D_t^U$", color=DARK_GRAY)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    # ============================================================
    # 5. Static annotation boxes
    # ============================================================
    csad_text = (
        f"CSAD regression\n"
        f"γ₂ = {gamma2:.4f}\n"
        f"t = {t_gamma2:.2f}\n"
        f"p = {p_gamma2:.3f}"
    )

    cssd_text = (
        f"CSSD regression, tail = {int(tail_q * 100)}%\n"
        f"βL = {beta_L:.4f}, pL = {p_beta_L:.3f}\n"
        f"βU = {beta_U:.4f}, pU = {p_beta_U:.3f}"
    )

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    fig.text(
        pos1.x0,
        pos1.y1 + 0.012,
        csad_text,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        bbox=dict(
            boxstyle="round",
            fc=(1, 1, 1, 0.78),
            ec="none",
        ),
    )

    fig.text(
        pos2.x0,
        pos2.y1 + 0.012,
        cssd_text,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        bbox=dict(
            boxstyle="round",
            fc=(1, 1, 1, 0.78),
            ec="none",
        ),
    )

    # ============================================================
    # 6. Artists for animation
    # ============================================================
    csad_line, = ax1.plot([], [], linewidth=1.8, color=LINE_GRAY)
    csad_dot = ax1.scatter([], [], s=70, color="red", zorder=5)

    cssd_bars = ax2.bar(
        cssd_reg["date"],
        np.zeros(len(cssd_reg)),
        width=2.0,
        color=CSSD_LIGHT_NAVY,
        alpha=0.85,
    )

    current_line1 = ax1.axvline(dates[0], color=LIGHTER_GRAY, lw=0.9, alpha=0.7)
    current_line2 = ax2.axvline(dates[0], color=LIGHTER_GRAY, lw=0.9, alpha=0.7)

    date_text = fig.text(
        0.5,
        0.009,
        "",
        ha="center",
        va="bottom",
        fontsize=12,
        color=DARK_GRAY,
        fontweight="bold",
    )

    #fig.suptitle(
        #"Regression-Based Herding Components Over Time",
       # fontsize=22,
      #  fontweight="bold",
     #   color=DARK_GRAY,
    #    y=0.98,
   # )

    fig.subplots_adjust(top=0.88, hspace=0.45)

    # ============================================================
    # 7. Save transparent frames
    # ============================================================
    frame_dir = base / OUT_REG_COMPONENTS_FRAME_DIR

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    frames_to_save = list(range(0, len(csad_reg), FRAME_STEP))

    if frames_to_save[-1] != len(csad_reg) - 1:
        frames_to_save.append(len(csad_reg) - 1)

    for out_i, frame in enumerate(frames_to_save):
        current_dates = csad_reg["date"].iloc[: frame + 1]
        current_values = csad_reg["csad_nonlinear_component"].iloc[: frame + 1]

        csad_line.set_data(current_dates, current_values)
        csad_dot.set_offsets([[dates_num[frame], csad_component[frame]]])

        for i, bar in enumerate(cssd_bars):
            if i <= frame:
                bar.set_height(cssd_component[i])
            else:
                bar.set_height(0)

        current_date = dates[frame]
        current_line1.set_xdata([current_date, current_date])
        current_line2.set_xdata([current_date, current_date])

        date_text.set_text(f"As of {current_date.strftime('%Y-%m-%d')}")

        fig.savefig(
            frame_dir / f"frame_{out_i:04d}.png",
            dpi=240,
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
        )

    print(f"Saved transparent regression-component frames -> {frame_dir}")

    # ============================================================
    # 8. Save transparent GIF
    # ============================================================
    gif_path = base / OUT_REG_COMPONENTS_GIF

    save_transparent_gif_from_frames(
        frame_dir=frame_dir,
        gif_path=gif_path,
        fps=fps,
        alpha_threshold=5,
    )

    print(f"Saved transparent regression-component GIF -> {gif_path}")
    print(f"GIF exists: {gif_path.exists()}")
    print(f"GIF size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    plt.close(fig)




def make_csad_gamma2_conceptual_animation(
    base: Path,
    fps: int = 12,
    n_frames: int = 120,
) -> None:
    """
    Conceptual transparent GIF:
    Shows how gamma_2 changes the CSAD curve shape.

    Positive gamma_2  -> dispersion accelerates upward
    Zero gamma_2      -> V-shape benchmark
    Negative gamma_2  -> downward tail-bending / herding intuition
    """

    frame_dir = base / OUT_GAMMA2_CONCEPT_FRAME_DIR
    gif_path = base / OUT_GAMMA2_CONCEPT_GIF

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    # Market return axis
    Rm = np.linspace(-15, 15, 500)

    # Conceptual CSAD model
    alpha = 1.0
    gamma1 = 0.50

    gamma2_positive = 0.020
    gamma2_zero = 0.000
    gamma2_negative = -0.020

    csad_positive = alpha + gamma1 * np.abs(Rm) + gamma2_positive * Rm**2
    csad_zero = alpha + gamma1 * np.abs(Rm) + gamma2_zero * Rm**2
    csad_negative = alpha + gamma1 * np.abs(Rm) + gamma2_negative * Rm**2

    # Make gamma_2 move positive -> negative -> positive
    gamma2_path = np.concatenate([
        np.linspace(gamma2_positive, gamma2_negative, n_frames // 2),
        np.linspace(gamma2_negative, gamma2_positive, n_frames // 2),
    ])

    GREEN = "#1B7F2A"
    BLUE = "#1F4FE0"
    RED = "#D60000"

    fig, ax = plt.subplots(figsize=(10.5, 6), facecolor="none")
    fig.patch.set_alpha(0)

    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.grid(False)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(LIGHTER_GRAY)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(axis="both", colors=MED_GRAY)

    ax.axvline(0, color=LIGHTER_GRAY, linewidth=1.0, alpha=0.8)
    ax.axhline(0, color=LIGHTER_GRAY, linewidth=1.0, alpha=0.4)

    # Reference curves
    ax.plot(
        Rm,
        csad_positive,
        color=GREEN,
        linestyle="-.",
        linewidth=1.8,
        alpha=0.45,
        label=r"$\gamma_2 > 0$: dispersion accelerates",
    )

    ax.plot(
        Rm,
        csad_zero,
        color=BLUE,
        linestyle="--",
        linewidth=1.8,
        alpha=0.45,
        label=r"$\gamma_2 = 0$: V-shape benchmark",
    )

    ax.plot(
        Rm,
        csad_negative,
        color=RED,
        linestyle="-",
        linewidth=1.8,
        alpha=0.45,
        label=r"$\gamma_2 < 0$: tails bend downward",
    )

    moving_line, = ax.plot([], [], linewidth=3.0, alpha=0.95)

    info_text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color=DARK_GRAY,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.78), ec="none"),
    )

    ax.set_xlim(-16, 16)
    ax.set_ylim(0, max(csad_positive) + 1.0)

    ax.set_xlabel(r"Actual Market Return $R_{m,t}$", color=DARK_GRAY)
    ax.set_ylabel("Cross-Sectional Return Dispersion (CSAD)", color=DARK_GRAY)

    ax.set_title(
        r"How the Nonlinear Term $\gamma_2 R_{m,t}^2$ Shapes CSAD",
        color=DARK_GRAY,
        fontweight="bold",
        pad=14,
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        frameon=True,
        fontsize=11,
    )

    legend.get_frame().set_alpha(0.75)
    legend.get_frame().set_edgecolor("none")

    fig.tight_layout()

    for i, gamma2 in enumerate(gamma2_path):
        csad_curve = alpha + gamma1 * np.abs(Rm) + gamma2 * Rm**2

        moving_line.set_data(Rm, csad_curve)

        if gamma2 > 0.003:
            state = "Dispersion accelerates"
            color = GREEN
        elif gamma2 < -0.003:
            state = "Tail-bending / herding intuition"
            color = RED
        else:
            state = "V-shape benchmark"
            color = BLUE

        moving_line.set_color(color)

        info_text.set_text(
            r"$CSAD_t = \alpha + \gamma_1 |R_{m,t}| + \gamma_2 R_{m,t}^2$"
            + "\n"
            + rf"$\gamma_2 = {gamma2:.4f}$"
            + "\n"
            + state
        )

        fig.savefig(
            frame_dir / f"frame_{i:04d}.png",
            dpi=220,
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
        )

    plt.close(fig)

    save_transparent_gif_from_frames(
        frame_dir=frame_dir,
        gif_path=gif_path,
        fps=fps,
        alpha_threshold=5,
    )

    print(f"Saved conceptual gamma2 transparent GIF -> {gif_path}")

##++

def main() -> None:
    base = project_dir()
    input_path = base / INPUT_FILE

    print(f"Loading cleaned panel -> {input_path}")
    panel = load_clean_panel(input_path)

    print("Building wide price matrix ...")
    prices_wide = build_wide_price_matrix(panel, style=WIDE_COLUMN_STYLE)

    print("Computing log returns ...")
    returns_wide = compute_log_returns(prices_wide)



    ##

    print("Creating conceptual gamma2 curve-morph transparent GIF ...")
    make_csad_gamma2_conceptual_animation(
        base=base,
        fps=12,
        n_frames=120,
    )
    ##

    print("Computing daily market return, CSAD, CSSD ...")
    daily_stats = compute_daily_dispersion(
        returns_wide,
        min_coins_per_day=MIN_COINS_PER_DAY,
    )

    if daily_stats.empty:
        raise RuntimeError(
            "No valid daily CSAD/CSSD rows were produced. "
            "Lower MIN_COINS_PER_DAY or inspect the cleaned dataset."
        )





    save_outputs(base, prices_wide, returns_wide, daily_stats)
    save_outputs(base, prices_wide, returns_wide, daily_stats)

    print("Creating CSAD/CSSD regression component plot ...")
    make_regression_component_plot(
        daily_stats=daily_stats,
        base=base,
        tail_q=0.05,
    )

    print("Creating top-5 log returns + CSAD GIF ...")

    top5_labels = top_n_labels_by_latest_market_cap(
        panel=panel,
        style=WIDE_COLUMN_STYLE,
        n=5,
    )

    make_top5_logreturns_csad_gif(
        returns_wide=returns_wide,
        daily_stats=daily_stats,
        base=base,
        top_coins=top5_labels,
        fps=10,
        target_duration_sec=20,
    )

    print("Creating transparent CSAD/CSSD regression-component GIF ...")
    make_regression_component_animation(
        daily_stats=daily_stats,
        base=base,
        tail_q=0.05,
        fps=10,
    )



    print("Creating static plots ...")
    make_static_plots(daily_stats, base)

    print("Creating UPDATED side-by-side CSAD/CSSD GIF animation ...")
    make_animation(daily_stats, base, event_labels=EVENT_LABELS)


    print("Creating log returns + CSAD figure ...")
    make_log_returns_and_csad_figure(
        returns_wide=returns_wide,
        daily_stats=daily_stats,
        base=base,
        event_labels=EVENT_LABELS,
    )

    print("Creating CSAD stacked transparent GIF ...")
    make_scatter_top_timeseries_bottom_animation(
        daily_stats=daily_stats,
        base=base,
        metric="CSAD",
        out_gif_name=OUT_CSAD_STACKED_GIF,
        out_frame_dir=OUT_CSAD_STACKED_FRAME_DIR,
        event_labels=EVENT_LABELS,
        show_event_labels=True,
    )

    print("Creating CSSD stacked transparent GIF ...")
    make_scatter_top_timeseries_bottom_animation(
        daily_stats=daily_stats,
        base=base,
        metric="CSSD",
        out_gif_name=OUT_CSSD_STACKED_GIF,
        out_frame_dir=OUT_CSSD_STACKED_FRAME_DIR,
        event_labels=EVENT_LABELS,
        show_event_labels=True,
    )

    highlight_labels = {
        # Put only dates that exist in your current dataset
        # "2025-08-05": "Some event",
        # "2025-11-12": "Another event",
    }



    print("\nDone.")
    print(f"Rows in cleaned panel: {len(panel)}")
    print(f"Rows in daily CSAD/CSSD table: {len(daily_stats)}")
    print(f"Date range: {daily_stats['date'].min()} -> {daily_stats['date'].max()}")
    print("\nFiles created:")
    print(f" - {OUT_WIDE_PRICES}")
    print(f" - {OUT_WIDE_RETURNS}")
    print(f" - {OUT_DAILY_STATS}")
    print(f" - {OUT_CSAD_PNG}")
    print(f" - {OUT_CSSD_PNG}")
    print(f" - {OUT_COMPARISON_PNG}")

    print(f" - {OUT_LOGRET_CSAD_FIG}")
    print(f" - {OUT_LOGRET_CSAD_CSV}")


    if shutil.which("ffmpeg") is not None:
        print(f" - {OUT_ANIM_MP4}")
    else:
        print(f" - {OUT_ANIM_GIF}")
        print(f" - {OUT_PROF_STYLE_HTML}")





if __name__ == "__main__":
    main()