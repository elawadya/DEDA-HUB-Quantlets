from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import shutil
from matplotlib.animation import FuncAnimation, PillowWriter

import subprocess
import shutil

import statsmodels.api as sm

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# =========================
# SETTINGS
# =========================
INPUT_FILE = "All_top_coins-updated-21.csv"

PRICE_COL = "Adj Close"   # use adjusted close price
COIN_COL = "Currency Name"
DATE_COL = "Date"

MIN_COINS_PER_DAY = 5

OUT_RETURNS_WIDE = "second_dataset_log_returns_wide.csv"
OUT_DAILY_CSAD = "second_dataset_daily_csad.csv"
OUT_PLOT = "second_dataset_log_returns_and_csad.png"
OUT_KAGGLE_REG_COMPONENTS_GIF = "kaggle_csad_cssd_regression_components_animation.gif"
OUT_KAGGLE_REG_COMPONENTS_FRAME_DIR = "kaggle_csad_cssd_regression_components_frames"


DARK_GRAY = "#666666"
MED_GRAY = "#8A8A8A"
LIGHTER_GRAY = "#B3B3B3"
LINE_GRAY = "#4F4F4F"
CSSD_LIGHT_NAVY = "#5B7DB1"


OUT_ANIM_MP4 = "second_dataset_csad_cssd_animation.mp4"
OUT_ANIM_GIF = "second_dataset_csad_cssd_animation.gif"
FPS = 12
DOT_SIZE = 28


# Important events shown on the plot
EVENT_WINDOWS = [
    {
        "start": "2017-12-01",
        "end": "2017-12-31",
        "label": "2017 crypto bubble peak",
    },
    {
        "start": "2020-03-08",
        "end": "2020-03-20",
        "label": "COVID market crash",
    },
]


HIGHLIGHT_COINS = {
    "Bitcoin": "tab:blue",
    "Ethereum": "tab:orange",
    "Solana": "tab:green",
    "Market Avg.": "black",
}


# =========================
# FUNCTIONS
# =========================
def project_dir() -> Path:
    return Path(__file__).resolve().parent



# =========================
# Helper FUNCTIONS
# =========================
def save_transparent_gif_from_frames(
    frame_dir: Path,
    gif_path: Path,
    fps: int = 10,
    alpha_threshold: int = 5,
) -> None:
    from PIL import Image

    frame_files = sorted(frame_dir.glob("frame_*.png"))

    if not frame_files:
        raise RuntimeError(f"No PNG frames found in {frame_dir}")

    frames = []

    for f in frame_files:
        img = Image.open(f).convert("RGBA")
        alpha = img.getchannel("A")

        paletted = img.convert("P", palette=Image.ADAPTIVE, colors=255)

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


def load_second_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required = {DATE_COL, PRICE_COL, COIN_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df[COIN_COL] = df[COIN_COL].astype(str).str.strip()

    df = df.dropna(subset=[DATE_COL, PRICE_COL, COIN_COL]).copy()
    df = df[df[PRICE_COL] > 0].copy()

    df = df.sort_values([COIN_COL, DATE_COL]).reset_index(drop=True)

    return df


def build_price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    prices = (
        df.pivot_table(
            index=DATE_COL,
            columns=COIN_COL,
            values=PRICE_COL,
            aggfunc="last",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    prices.index.name = "date"
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices) - np.log(prices.shift(1))
    returns.index.name = "date"
    return returns


def compute_csad(returns: pd.DataFrame) -> pd.DataFrame:
    n_coins = returns.notna().sum(axis=1)

    market_return = returns.mean(axis=1, skipna=True)
    deviations = returns.sub(market_return, axis=0)

    csad = deviations.abs().mean(axis=1, skipna=True)
    cssd = deviations.std(axis=1, ddof=1, skipna=True)

    out = pd.DataFrame({
        "date": returns.index,
        "market_return": market_return,
        "CSAD": csad,
        "CSSD": cssd,
        "n_coins": n_coins,
    })

    out = out[out["n_coins"] >= MIN_COINS_PER_DAY].copy()
    out = out.dropna(subset=["market_return", "CSAD", "CSSD"]).reset_index(drop=True)

    return out


def make_log_return_csad_plot(
    returns: pd.DataFrame,
    csad: pd.DataFrame,
    base: Path,
) -> None:
    stats = csad.copy()
    stats["date"] = pd.to_datetime(stats["date"])
    stats = stats.set_index("date").sort_index()

    ret = returns.copy()
    ret.index = pd.to_datetime(ret.index)
    ret = ret.sort_index()

    common_idx = ret.index.intersection(stats.index)
    data_start = common_idx.min()
    data_end = common_idx.max()

    # keep only events inside the dataset period
    valid_events = []
    for ev in valid_events:
        start = pd.to_datetime(ev["start"])
        end = pd.to_datetime(ev["end"])

        if start <= data_end and end >= data_start:
            valid_events.append(ev)
    ret = ret.loc[common_idx]
    stats = stats.loc[common_idx]

    # Choose available highlighted coins
    chosen = [c for c in HIGHLIGHT_COINS.keys() if c in ret.columns]

    # If too few highlighted coins exist, add first available coins
    if len(chosen) < 5:
        extras = [c for c in ret.columns if c not in chosen]
        chosen.extend(extras[: max(0, 5 - len(chosen))])

    ret_plot = ret[chosen].copy()
    ret_plot["Market Avg."] = stats["market_return"]

    combined = ret_plot.copy()
    combined["CSAD"] = stats["CSAD"]

    combined["market_return"] = ret_plot.mean(axis=1, skipna=True)
    combined.to_csv(base / "second_dataset_log_returns_and_csad_data.csv")





    ##

    # =========================
    # HERDING REGRESSION TEST
    # =========================
    import statsmodels.api as sm

    df_reg = combined.copy()

    # Check your column names first
    print(df_reg.columns)

    # Use the correct market return column name from your combined dataframe
    # Common possibilities: "market_return", "Rm", "r_m", "mean_return"
    MARKET_RETURN_COL = "market_return"  # change this if your column has another name
    CSAD_COL = "CSAD"

    df_reg["abs_Rm"] = df_reg[MARKET_RETURN_COL].abs()
    df_reg["sq_Rm"] = df_reg[MARKET_RETURN_COL] ** 2

    df_reg = df_reg.dropna(subset=[CSAD_COL, "abs_Rm", "sq_Rm"])

    y = df_reg[CSAD_COL]
    X = df_reg[["abs_Rm", "sq_Rm"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    print(model.summary())

    with open(base / "second_dataset_csad_herding_regression_summary.txt", "w") as f:
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




    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        sharex=True,
        facecolor=(0, 0, 0, 0),
    )

    fig.patch.set_alpha(0)

    for ax in [ax1, ax2]:
        ax.set_facecolor((0, 0, 0, 0))
        ax.patch.set_alpha(0)

    line_colors = {}

    for col in ret_plot.columns:
        if col in HIGHLIGHT_COINS:
            color = HIGHLIGHT_COINS[col]
            lw = 2.2 if col == "Market Avg." else 1.6
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
            label=col,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )

    ax1.set_title("Log Returns and CSAD with Important Crypto Events", fontsize=18)
    ax1.set_ylabel("Log return")
    ax1.grid(alpha=0.25)

    ax2.plot(stats.index, stats["CSAD"], color="black", linewidth=1.3)
    ax2.set_ylabel("CSAD")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.25)

    # Add event windows
    for ev in EVENT_WINDOWS:
        start = pd.to_datetime(ev["start"])
        end = pd.to_datetime(ev["end"])
        label = ev["label"]

        for ax in [ax1, ax2]:
            ax.axvspan(start, end, alpha=0.12)

        mid = start + (end - start) / 2
        ax1.text(
            mid,
            ax1.get_ylim()[1],
            label,
            rotation=90,
            fontsize=9,
            va="top",
            ha="center",
        )

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right")

    ax1.legend(loc="upper right", fontsize=9, ncol=2)

    sample_text = f"Sample period: {common_idx.min().date()} - {common_idx.max().date()}"
    fig.text(0.08, 0.02, sample_text, fontsize=10)

    ax1.set_xlim(data_start, data_end)
    ax2.set_xlim(data_start, data_end)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    fig.savefig(
        base / "second_dataset_log_returns_and_csad_transparent.png",
        dpi=220,
        bbox_inches="tight",
        transparent=True,
        facecolor=(0, 0, 0, 0),
        edgecolor=(0, 0, 0, 0),
    )



    plt.close(fig)

    print(f"Saved plot: {base / OUT_PLOT}")
    print("Saved figure data: second_dataset_log_returns_and_csad_data.csv")

    def make_csad_cssd_animation(csad: pd.DataFrame, base: Path) -> None:
        df = csad.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        x = df["market_return"].to_numpy()
        y_csad = df["CSAD"].to_numpy()
        y_cssd = df["CSSD"].to_numpy()
        dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

        x_min, x_max = np.nanmin(x), np.nanmax(x)
        csad_min, csad_max = np.nanmin(y_csad), np.nanmax(y_csad)
        cssd_min, cssd_max = np.nanmin(y_cssd), np.nanmax(y_cssd)

        x_pad = max((x_max - x_min) * 0.08, 1e-6)
        csad_pad = max((csad_max - csad_min) * 0.08, 1e-6)
        cssd_pad = max((cssd_max - cssd_min) * 0.08, 1e-6)

        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            facecolor=(0, 0, 0, 0),
        )

        fig.subplots_adjust(bottom=0.18)

        ax1.set_xlim(x_min - x_pad, x_max + x_pad)
        ax1.set_ylim(csad_min - csad_pad, csad_max + csad_pad)
        ax1.set_xlabel("Market return")
        ax1.set_ylabel("CSAD")
        ax1.set_title("CSAD vs Market Return")
        ax1.grid(alpha=0.25)

        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(cssd_min - cssd_pad, cssd_max + cssd_pad)
        ax2.set_xlabel("Market return")
        ax2.set_ylabel("CSSD")
        ax2.set_title("CSSD vs Market Return")
        ax2.grid(alpha=0.25)

        scat1 = ax1.scatter([], [], s=DOT_SIZE)
        scat2 = ax2.scatter([], [], s=DOT_SIZE)

        date_text = fig.text(
            0.5, 0.04, "",
            ha="center",
            fontsize=14,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4)
        )

        def update(frame: int):
            xs = x[: frame + 1]

            scat1.set_offsets(np.column_stack([xs, y_csad[: frame + 1]]))
            scat2.set_offsets(np.column_stack([xs, y_cssd[: frame + 1]]))

            date_text.set_text(
                f"Date: {dates[frame]} | N coins: {int(df.loc[frame, 'n_coins'])}"
            )

            return scat1, scat2, date_text

        anim = FuncAnimation(
            fig,
            update,
            frames=len(df),
            interval=1000 / FPS,
            blit=False,
        )

        mp4_path = base / OUT_ANIM_MP4
        gif_path = base / OUT_ANIM_GIF

        if shutil.which("ffmpeg") is not None:
            writer = FFMpegWriter(fps=FPS, bitrate=2400)
            anim.save(mp4_path, writer=writer, dpi=180)
            print(f"Saved MP4 animation: {mp4_path}")
        else:
            writer = PillowWriter(fps=FPS)
            anim.save(gif_path, writer=writer, dpi=140)
            print(f"ffmpeg not found. Saved GIF instead: {gif_path}")

        plt.close(fig)



def make_csad_cssd_animation(csad: pd.DataFrame, base: Path) -> None:
    df = csad.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    x = df["market_return"].to_numpy()
    y_csad = df["CSAD"].to_numpy()
    y_cssd = df["CSSD"].to_numpy()
    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    csad_min, csad_max = np.nanmin(y_csad), np.nanmax(y_csad)
    cssd_min, cssd_max = np.nanmin(y_cssd), np.nanmax(y_cssd)

    x_pad = max((x_max - x_min) * 0.08, 1e-6)
    csad_pad = max((csad_max - csad_min) * 0.08, 1e-6)
    cssd_pad = max((cssd_max - cssd_min) * 0.08, 1e-6)

    frame_dir = base / "kaggle_transparent_frames"

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        facecolor=(0, 0, 0, 0),
    )

    fig.patch.set_alpha(0)

    for ax in [ax1, ax2]:
        ax.set_facecolor((0, 0, 0, 0))
        ax.patch.set_alpha(0)
        ax.grid(False)

    ax1.set_xlim(x_min - x_pad, x_max + x_pad)
    ax1.set_ylim(csad_min - csad_pad, csad_max + csad_pad)
    ax1.set_xlabel("Market return")
    ax1.set_ylabel("CSAD")
    ax1.set_title("CSAD vs Market Return")

    ax2.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_ylim(cssd_min - cssd_pad, cssd_max + cssd_pad)
    ax2.set_xlabel("Market return")
    ax2.set_ylabel("CSSD")
    ax2.set_title("CSSD vs Market Return")

    cmap = plt.colormaps["turbo"].resampled(len(df))
    point_colors = [cmap(i) for i in range(len(df))]

    scat1 = ax1.scatter([], [], s=28)
    scat2 = ax2.scatter([], [], s=28)

    date_text = ax1.text(
        0.02, 0.96, "",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3),
    )

    def update(frame: int):
        xs = x[: frame + 1]
        colors = point_colors[: frame + 1]

        scat1.set_offsets(np.column_stack([xs, y_csad[: frame + 1]]))
        scat1.set_facecolors(colors)
        scat1.set_edgecolors(colors)

        scat2.set_offsets(np.column_stack([xs, y_cssd[: frame + 1]]))
        scat2.set_facecolors(colors)
        scat2.set_edgecolors(colors)

        date_text.set_text(
            f"Date: {dates[frame]} | N coins: {int(df.loc[frame, 'n_coins'])}"
        )

    # Save every 5th frame to make it faster
    frames = range(0, len(df), 5)

    for i, frame in enumerate(frames):
        update(frame)

        fig.savefig(
            frame_dir / f"frame_{i:04d}.png",
            dpi=160,
            transparent=True,
            facecolor=(0, 0, 0, 0),
            edgecolor=(0, 0, 0, 0),
            bbox_inches="tight",
            pad_inches=0.05,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=range(0, len(df), 5),
        interval=1000 / 12,
        blit=False,
    )

    gif_path = base / "kaggle_csad_cssd_transparent_keynote.gif"

    writer = PillowWriter(fps=12)
    anim.save(
        gif_path,
        writer=writer,
        dpi=160,
        savefig_kwargs={
            "transparent": True,
            "facecolor": (0, 0, 0, 0),
            "edgecolor": (0, 0, 0, 0),
        },
    )

    plt.close(fig)
    print(f"Saved transparent GIF for Keynote: {gif_path}")

    def update(frame: int):
        xs = x[: frame + 1]
        colors = point_colors[: frame + 1]

        scat1.set_offsets(np.column_stack([xs, y_csad[: frame + 1]]))
        scat1.set_facecolors(colors)
        scat1.set_edgecolors(colors)

        scat2.set_offsets(np.column_stack([xs, y_cssd[: frame + 1]]))
        scat2.set_facecolors(colors)
        scat2.set_edgecolors(colors)
        date_text.set_text(f"Date: {dates[frame]} | N coins: {int(df.loc[frame, 'n_coins'])}")
        return scat1, scat2, date_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000 / 12,
        blit=False,
    )

    mp4_path = base / "second_dataset_csad_cssd_animation.mp4"
    gif_path = base / "second_dataset_csad_cssd_animation.gif"

    if shutil.which("ffmpeg") is not None:
        writer = FFMpegWriter(fps=12, bitrate=2400)
        anim.save(mp4_path, writer=writer, dpi=180)
        print(f"Saved MP4 animation: {mp4_path}")
    else:
        writer = PillowWriter(fps=12)
        anim.save(gif_path, writer=writer, dpi=140)
        print(f"ffmpeg not found. Saved GIF instead: {gif_path}")

    plt.close(fig)




    ##

def make_kaggle_regression_component_animation(
    daily_stats: pd.DataFrame,
    base: Path,
    tail_q: float = 0.05,
    fps: int = 10,
) -> None:
    """
    Transparent GIF for Kaggle dataset.

    Top panel:
        CSAD nonlinear component = gamma_2 * R_m,t^2

    Bottom panel:
        CSSD extreme-market component = beta_L * D_L + beta_U * D_U
    """

    import statsmodels.api as sm

    df = daily_stats.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------
    # CSAD regression
    # CSAD_t = alpha + gamma1 |Rm,t| + gamma2 Rm,t^2 + error
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # CSSD regression
    # CSSD_t = alpha + beta_L D_L + beta_U D_U + error
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Sync data lengths / dates
    # ------------------------------------------------------------
    common_dates = pd.Index(csad_reg["date"]).intersection(pd.Index(cssd_reg["date"]))
    csad_reg = csad_reg[csad_reg["date"].isin(common_dates)].copy().reset_index(drop=True)
    cssd_reg = cssd_reg[cssd_reg["date"].isin(common_dates)].copy().reset_index(drop=True)

    dates = csad_reg["date"].tolist()
    dates_num = mdates.date2num(dates)

    csad_component = csad_reg["csad_nonlinear_component"].to_numpy()
    cssd_component = cssd_reg["cssd_extreme_component"].to_numpy()

    # ------------------------------------------------------------
    # Axis ranges
    # ------------------------------------------------------------
    y1_min = min(0, np.nanmin(csad_component))
    y1_max = max(0, np.nanmax(csad_component))
    y1_pad = max((y1_max - y1_min) * 0.15, 1e-6)

    y2_min = min(0, np.nanmin(cssd_component))
    y2_max = max(0, np.nanmax(cssd_component))
    y2_pad = max((y2_max - y2_min) * 0.15, 1e-6)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        facecolor="none",
    )

    fig.patch.set_alpha(0)
    fig.subplots_adjust(
        top=0.88,
        bottom=0.16,
        left=0.10,
        right=0.95,
        hspace=0.50,
    )

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

    ax1.set_xlim(min(dates), max(dates))
    ax2.set_xlim(min(dates), max(dates))

    ax1.set_ylim(y1_min - y1_pad, y1_max + y1_pad)
    ax2.set_ylim(y2_min - y2_pad, y2_max + y2_pad)

    ax1.set_title(
        "CSAD Nonlinear Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
        pad=22,
    )

    ax2.set_title(
        "CSSD Extreme-Market Component Over Time",
        fontsize=18,
        fontweight="bold",
        color=DARK_GRAY,
        pad=22,
    )

    ax1.set_ylabel(r"$\gamma_2 R_{m,t}^2$", color=DARK_GRAY)
    ax2.set_ylabel(r"$\beta_L D_t^L + \beta_U D_t^U$", color=DARK_GRAY)

    # Kaggle dataset covers many years, so show years only
    ax2.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(
        ax2.get_xticklabels(),
        rotation=0,
        ha="center",
        color=DARK_GRAY,
    )



    # ------------------------------------------------------------
    # Regression summary text
    # ------------------------------------------------------------
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

    # Put boxes OUTSIDE plot borders using fig.text
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    fig.text(
        pos1.x0 - 0.01,
        pos1.y1 + 0.010,
        csad_text,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.78), ec="none"),
    )

    fig.text(
        pos2.x0 - 0.01,
        pos2.y1 + 0.010,
        cssd_text,
        ha="left",
        va="bottom",
        fontsize=11,
        color=DARK_GRAY,
        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.78), ec="none"),
    )
    # ------------------------------------------------------------
    # Artists
    # ------------------------------------------------------------
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
        0.60,
        0.005,
        "",
        ha="center",
        va="bottom",
        fontsize=12,
        color=DARK_GRAY,
        fontweight="bold",
    )

    # ------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------
    frame_dir = base / OUT_KAGGLE_REG_COMPONENTS_FRAME_DIR

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    FRAME_STEP_LOCAL = 2
    frames_to_save = list(range(0, len(csad_reg), FRAME_STEP_LOCAL))
    if frames_to_save[-1] != len(csad_reg) - 1:
        frames_to_save.append(len(csad_reg) - 1)

    for out_i, frame in enumerate(frames_to_save):
        current_dates = csad_reg["date"].iloc[: frame + 1]
        current_values = csad_reg["csad_nonlinear_component"].iloc[: frame + 1]

        # top line
        csad_line.set_data(current_dates, current_values)
        csad_dot.set_offsets([[dates_num[frame], csad_component[frame]]])

        # bottom bars
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

    print(f"Saved transparent Kaggle regression frames -> {frame_dir}")

    # ------------------------------------------------------------
    # GIF
    # ------------------------------------------------------------
    gif_path = base / OUT_KAGGLE_REG_COMPONENTS_GIF

    save_transparent_gif_from_frames(
        frame_dir=frame_dir,
        gif_path=gif_path,
        fps=fps,
        alpha_threshold=5,
    )

    print(f"Saved transparent Kaggle regression GIF -> {gif_path}")
    print(f"GIF exists: {gif_path.exists()}")
    print(f"GIF size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Optional: save regression summaries
    with open(base / "kaggle_regression_components_summary.txt", "w") as f:
        f.write("CSAD regression:\n")
        f.write(csad_model.summary().as_text())
        f.write("\n\nCSSD regression:\n")
        f.write(cssd_model.summary().as_text())

    plt.close(fig)

##
def main() -> None:
    base = project_dir()
    input_path = base / INPUT_FILE

    print(f"Loading: {input_path}")
    df = load_second_dataset(input_path)

    print("Building wide price matrix...")
    prices = build_price_matrix(df)

    print("Computing log returns...")
    returns = compute_log_returns(prices)

    print("Computing CSAD...")
    csad = compute_csad(returns)

    returns.to_csv(base / OUT_RETURNS_WIDE)
    csad.to_csv(base / OUT_DAILY_CSAD, index=False)

    print("Creating log return + CSAD plot...")
    make_log_return_csad_plot(returns, csad, base)

    print("Creating CSAD/CSSD animation video...")
    make_csad_cssd_animation(csad, base)

    print("Creating transparent Kaggle CSAD/CSSD regression-component GIF ...")
    make_kaggle_regression_component_animation(
        daily_stats=csad,
        base=base,
        tail_q=0.05,
        fps=10,
    )

    print("\nDone.")
    print(f"Coins: {prices.shape[1]}")
    print(f"Days: {len(csad)}")
    print(f"Date range: {csad['date'].min()} -> {csad['date'].max()}")


if __name__ == "__main__":
    main()
