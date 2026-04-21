"""
Publication-quality plots for the UAE CPI project.

Style: BCG/McKinsey slide aesthetic — clean, light background, strong typography,
annotated callouts. Exports PNG at 180 DPI for LinkedIn and the GitHub README.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Consulting palette (navy, accent, muted)
PALETTE = {
    "navy": "#0B3D5A",
    "teal": "#1E88A8",
    "accent": "#E07A1A",
    "red": "#C0392B",
    "green": "#2F8F5F",
    "grey": "#6B7280",
    "grid": "#E5E7EB",
    "bg": "#FAFAF7",
}

BASE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": PALETTE["grey"],
    "axes.labelcolor": "#1F2937",
    "axes.titleweight": "semibold",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.color": PALETTE["grey"],
    "ytick.color": PALETTE["grey"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": "#1F2937",
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

plt.rcParams.update(BASE_RC)


def _annotate_events(ax, y_pos=None):
    """Shade major UAE inflation events."""
    events = [
        ("2015-11-01", "2016-12-31", PALETTE["accent"], "Oil crash\n(deflationary pressure)"),
        ("2018-01-01", "2018-12-31", PALETTE["red"], "5% VAT\nintroduced"),
        ("2020-03-01", "2021-06-30", PALETTE["teal"], "COVID-19\n& recovery"),
        ("2023-01-01", "2023-12-31", PALETTE["green"], "Global\ninflation spike"),
    ]
    y = y_pos if y_pos is not None else ax.get_ylim()[1] * 0.9
    for start, end, color, label in events:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=0.07, zorder=0)


def plot_headline_cpi(chained: pd.DataFrame, outpath: str):
    """Hero chart: 12 years of UAE headline CPI inflation with annotated regimes."""
    df = chained[chained["Date"] >= "2014-01-01"].copy()
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axhline(0, color=PALETTE["grey"], linewidth=0.8, zorder=1)
    ax.plot(df["Date"], df["YoY"], color=PALETTE["navy"], linewidth=2.2, zorder=3)
    ax.fill_between(df["Date"], 0, df["YoY"], where=df["YoY"] > 0, color=PALETTE["navy"], alpha=0.12, zorder=2)
    ax.fill_between(df["Date"], 0, df["YoY"], where=df["YoY"] <= 0, color=PALETTE["red"], alpha=0.12, zorder=2)

    _annotate_events(ax)

    # Peak callouts — nudged to avoid title overlap
    peak = df.loc[df["YoY"].idxmax()]
    trough = df.loc[df["YoY"].idxmin()]
    ax.set_ylim(-3.5, 10.5)
    ax.annotate(
        f"Peak: {peak['YoY']:.1f}%  ({peak['Date'].strftime('%b %Y')})",
        xy=(peak["Date"], peak["YoY"]), xytext=(peak["Date"] - pd.DateOffset(months=24), peak["YoY"] + 1.1),
        fontsize=9, color=PALETTE["navy"], fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=0.8),
    )
    ax.annotate(
        f"Trough: {trough['YoY']:.1f}%  ({trough['Date'].strftime('%b %Y')})",
        xy=(trough["Date"], trough["YoY"]), xytext=(trough["Date"] + pd.DateOffset(months=5), trough["YoY"] - 0.5),
        fontsize=9, color=PALETTE["red"], fontweight="semibold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.8),
    )

    ax.grid(True, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Headline CPI, YoY (%)", fontsize=10)
    ax.set_title(
        "UAE Inflation, 2014–2025: four macro eras, nine micro-regimes in twelve years",
        loc="left", pad=15,
    )
    fig.text(0.125, 0.02, "Source: FCSA · Author's calculations (chained 2014 + 2021 base indices)",
             fontsize=8, color=PALETTE["grey"])
    fig.savefig(outpath)
    plt.close(fig)


def plot_stl(chained: pd.DataFrame, stl_result, outpath: str):
    """STL trend/seasonal/residual panels with variance shares."""
    df = chained.set_index("Date")["YoY"].dropna()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df.index, df.values, color=PALETTE["navy"], lw=1.6)
    axes[0].set_title("Headline CPI YoY — STL decomposition", loc="left")
    axes[0].set_ylabel("Observed (%)")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(stl_result["trend"].index, stl_result["trend"].values, color=PALETTE["teal"], lw=1.8)
    axes[1].set_ylabel(f"Trend ({stl_result['trend_share']:.0%})")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(stl_result["seasonal"].index, stl_result["seasonal"].values, color=PALETTE["accent"], lw=1.2)
    axes[2].set_ylabel(f"Seasonal ({stl_result['seasonal_share']:.1%})")
    axes[2].grid(True, alpha=0.4)

    axes[3].plot(stl_result["resid"].index, stl_result["resid"].values, color=PALETTE["grey"], lw=1.0)
    axes[3].axhline(0, color=PALETTE["grey"], lw=0.5)
    axes[3].set_ylabel(f"Residual ({stl_result['residual_share']:.1%})")
    axes[3].set_xlabel("Year")
    axes[3].grid(True, alpha=0.4)

    axes[3].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.125, 0.01,
             "Takeaway: Trend explains the majority of variance — UAE inflation is structural, not seasonal.",
             fontsize=9, color=PALETTE["grey"], style="italic")
    fig.savefig(outpath)
    plt.close(fig)


def plot_sector_contributions(contrib: pd.DataFrame, outpath: str, start="2021-01-01"):
    """Stacked bar: how much each sector contributes to headline YoY over time."""
    d = contrib[contrib.index >= start].copy()
    cols_to_plot = [c for c in d.columns if not c.startswith("__")]
    # Force a stable, narrative order: Housing first (biggest weight), then by mean contribution magnitude
    priority = ["Housing/Utilities", "Transportation", "Food & Beverages",
                "Restaurants/Hotels", "Recreation/Culture", "Miscellaneous"]
    top = [c for c in priority if c in cols_to_plot]
    other = [c for c in cols_to_plot if c not in top]
    d["Other"] = d[other].sum(axis=1)
    plot_cols = top + ["Other"]

    colors = [PALETTE["navy"], PALETTE["teal"], PALETTE["accent"],
              PALETTE["red"], PALETTE["green"], "#8E44AD", PALETTE["grey"]][:len(plot_cols)]

    fig, ax = plt.subplots(figsize=(13, 6))
    pos = d[plot_cols].clip(lower=0)
    neg = d[plot_cols].clip(upper=0)
    pos_bottom = np.zeros(len(d))
    neg_bottom = np.zeros(len(d))
    width = 25

    for col, color in zip(plot_cols, colors):
        ax.bar(d.index, pos[col], bottom=pos_bottom, width=width, color=color, label=col, edgecolor="white", linewidth=0.2)
        ax.bar(d.index, neg[col], bottom=neg_bottom, width=width, color=color, edgecolor="white", linewidth=0.2)
        pos_bottom = pos_bottom + pos[col].values
        neg_bottom = neg_bottom + neg[col].values

    ax.plot(d.index, d["__headline"], color="black", lw=1.8, label="Headline YoY", zorder=5)
    ax.axhline(0, color=PALETTE["grey"], lw=0.8)
    ax.set_ylabel("Contribution to YoY (pp)", fontsize=10)
    ax.set_title("Composition of UAE inflation: Housing is the pivot, Transportation the amplifier",
                 loc="left", pad=12)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.4, axis="y")
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0, -0.12))
    fig.text(0.125, -0.09,
             "Housing (34% basket weight) carries the persistent positive signal; Transportation drove the 2022 oil-led spike.",
             fontsize=8, color=PALETTE["grey"])
    fig.savefig(outpath)
    plt.close(fig)


def plot_regime_map(umap_coords: pd.DataFrame, labels: pd.Series, outpath: str):
    """UMAP 2-D scatter colored by HDBSCAN regime, with temporal arrows."""
    fig, ax = plt.subplots(figsize=(11, 7))
    data = umap_coords.copy()
    data["regime"] = labels

    # Color by regime
    unique_labels = sorted([l for l in data["regime"].unique() if l >= 0])
    cmap = plt.get_cmap("tab10", max(10, len(unique_labels)))

    # Temporal path
    ax.plot(data["umap_1"], data["umap_2"], color=PALETTE["grey"], lw=0.5, alpha=0.4, zorder=1)

    # Points: color by regime, size by recency
    for i, lab in enumerate(unique_labels):
        sub = data[data["regime"] == lab]
        if sub.empty: continue
        label_text = f"Regime {lab}  ({sub.index.min().strftime('%b%y')} – {sub.index.max().strftime('%b%y')})"
        ax.scatter(sub["umap_1"], sub["umap_2"],
                   color=cmap(i), s=80, alpha=0.85, edgecolor="white", linewidth=0.8,
                   label=label_text, zorder=3)
    noise = data[data["regime"] == -1]
    if not noise.empty:
        ax.scatter(noise["umap_1"], noise["umap_2"],
                   color=PALETTE["grey"], s=30, alpha=0.4, marker="x",
                   label=f"Transitional ({len(noise)} months)", zorder=2)

    # Highlight current state
    latest = data.index.max()
    ax.scatter(data.loc[latest, "umap_1"], data.loc[latest, "umap_2"],
               s=250, facecolors="none", edgecolor=PALETTE["red"], linewidth=2.5, zorder=10)
    ax.annotate(f"Current: {latest.strftime('%b %Y')}",
                xy=(data.loc[latest, "umap_1"], data.loc[latest, "umap_2"]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=10, fontweight="bold", color=PALETTE["red"])

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Unsupervised discovery of UAE inflation regimes", loc="left", pad=12)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.text(0.125, 0.02,
             "HDBSCAN clustering of 13-sector z-score profiles, 2015–2025. "
             "Each point is one month; proximity = similar inflation composition.",
             fontsize=8, color=PALETTE["grey"])
    fig.savefig(outpath)
    plt.close(fig)


def plot_forecast_backtest(backtest_df: pd.DataFrame, outpath: str):
    """Model comparison: RMSE and directional accuracy across horizons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    color_map = {"naive_seasonal": PALETTE["grey"], "ridge": PALETTE["navy"], "xgboost": PALETTE["accent"]}
    pivot_rmse = backtest_df.pivot(index="horizon", columns="model", values="rmse")
    pivot_da = backtest_df.pivot(index="horizon", columns="model", values="directional_accuracy")

    # RMSE bars
    x = np.arange(len(pivot_rmse.index))
    width = 0.28
    for i, col in enumerate(["naive_seasonal", "ridge", "xgboost"]):
        if col in pivot_rmse.columns:
            ax1.bar(x + i*width - width, pivot_rmse[col], width,
                    color=color_map[col], label=col.replace("_", " ").title(),
                    edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h}-month" for h in pivot_rmse.index])
    ax1.set_ylabel("Out-of-sample RMSE (pp)")
    ax1.set_title("Forecast error: lower is better", loc="left")
    ax1.grid(True, alpha=0.4, axis="y")
    ax1.legend(frameon=False, fontsize=9)

    # Directional accuracy
    for i, col in enumerate(["naive_seasonal", "ridge", "xgboost"]):
        if col in pivot_da.columns:
            ax2.bar(x + i*width - width, pivot_da[col]*100, width,
                    color=color_map[col], edgecolor="white", linewidth=0.5)
    ax2.axhline(50, color=PALETTE["red"], lw=1, ls="--", alpha=0.6, label="Coin flip")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h}-month" for h in pivot_da.index])
    ax2.set_ylabel("Directional accuracy (%)")
    ax2.set_title("Got the direction right — up or down?", loc="left")
    ax2.grid(True, alpha=0.4, axis="y")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("Walk-forward backtest: 71 expanding-window origins, 2020–2025",
                 fontsize=12, fontweight="semibold", x=0.125, ha="left", y=1.02)
    fig.savefig(outpath)
    plt.close(fig)


def plot_forecast_timeline(predictions: pd.Series, actuals: pd.Series, naive_preds: pd.Series, outpath: str, horizon: int):
    """Time-series plot showing forecast vs actual vs naive."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    common = actuals.index
    ax.plot(common, actuals.values, color=PALETTE["navy"], lw=2, label="Actual headline YoY", zorder=3)
    ax.plot(predictions.index, predictions.values, color=PALETTE["accent"], lw=1.8,
            label=f"Ridge forecast ({horizon}-month ahead)", zorder=4, alpha=0.9)
    if naive_preds is not None:
        ax.plot(naive_preds.index, naive_preds.values, color=PALETTE["grey"],
                lw=1.2, ls="--", label="Naive seasonal baseline", zorder=2, alpha=0.8)
    ax.axhline(0, color=PALETTE["grey"], lw=0.6)
    ax.set_ylabel("YoY (%)")
    ax.set_title(f"Out-of-sample forecast performance, {horizon}-month horizon", loc="left", pad=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(outpath)
    plt.close(fig)


def plot_granger_bars(granger_df: pd.DataFrame, outpath: str):
    """Bar chart of -log10(p) from Granger causality tests."""
    d = granger_df.copy()
    d["neg_log_p"] = -np.log10(d["min_p"].clip(lower=1e-6))
    d = d.sort_values("neg_log_p", ascending=True)
    colors = [PALETTE["navy"] if s else PALETTE["grey"] for s in d["significant"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(d["sector"], d["neg_log_p"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(-np.log10(0.05), color=PALETTE["red"], ls="--", lw=1, label="p = 0.05 threshold")

    # Annotate best lag
    for bar, lag, p in zip(bars, d["best_lag_months"], d["min_p"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"lag {int(lag)}m  (p={p:.3f})",
                va="center", fontsize=8, color=PALETTE["grey"])

    ax.set_xlabel("Granger predictive strength  (–log₁₀ p)")
    ax.set_title("Which sectors predict headline inflation?", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.4, axis="x")
    fig.text(0.125, 0.01,
             "MoM YoY series, lags 1–6. Significant sectors (blue) lead headline inflation statistically.",
             fontsize=8, color=PALETTE["grey"])
    fig.savefig(outpath)
    plt.close(fig)
