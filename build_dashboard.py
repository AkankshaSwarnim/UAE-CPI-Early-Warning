"""
Build a standalone interactive HTML dashboard.

Output: docs/dashboard.html — self-contained, renders in any browser,
no server or installation needed. Can be hosted via GitHub Pages.

Tabs:
  1. Headline CPI with regime overlay
  2. Sector contributions (stacked)
  3. Regime map (UMAP scatter, hover for date/regime)
  4. Forecast vs actual (timeline with naive baseline)
  5. Model benchmark (side-by-side RMSE and directional accuracy)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_raw, build_chained_index, build_sector_panel, yoy_panel, mom_panel, CPI_WEIGHTS
from analysis import stl_variance_share, regime_features, hdbscan_regimes, umap_project, contribution_to_headline
from forecast import build_feature_matrix, make_ridge, walk_forward_backtest, naive_seasonal_backtest, run_full_benchmark


NAVY = "#0B3D5A"
TEAL = "#1E88A8"
ACCENT = "#E07A1A"
RED = "#C0392B"
GREEN = "#2F8F5F"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"


def build_dashboard(out_path: str = "docs/dashboard.html"):
    # ---- Data ----
    df = load_raw("data")
    chained = build_chained_index(df, "ALL")
    chained = chained[chained["Date"] >= "2014-01-01"].reset_index(drop=True)
    panel = build_sector_panel(df, start="2014-01-01")
    yoy = yoy_panel(panel)
    mom = mom_panel(panel)

    # Regimes
    feats = regime_features(yoy)
    labels = hdbscan_regimes(feats, min_cluster_size=6)
    coords = umap_project(feats)

    # Contributions
    contrib = contribution_to_headline(yoy, CPI_WEIGHTS)

    # Forecast
    X, y = build_feature_matrix(yoy, horizon=3)
    origin = yoy["All Items"].shift(3)
    ridge_bt = walk_forward_backtest(X, y, make_ridge, "ridge", min_train=48, origin_yoy=origin)
    naive_bt = naive_seasonal_backtest(yoy["All Items"], y.index[48:], horizon=3)

    # Benchmark
    bench = run_full_benchmark(yoy, horizons=(1, 3, 6))

    # ---- Plotly figure: 5 tabs via subplot with dropdown ----
    # Easier: build five separate figures, concatenate into one HTML with anchor nav

    figures = []

    # --- FIG 1: Headline CPI with regime shading ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=chained["Date"], y=chained["YoY"],
        mode="lines", name="Headline YoY",
        line=dict(color=NAVY, width=2.5),
        fill="tozeroy", fillcolor="rgba(11,61,90,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>YoY: %{y:.2f}%<extra></extra>",
    ))
    events = [
        ("2015-11-01", "2016-12-31", "Oil crash"),
        ("2018-01-01", "2018-12-31", "VAT introduced"),
        ("2020-03-01", "2021-06-30", "COVID & recovery"),
        ("2023-01-01", "2023-12-31", "Global inflation spike"),
    ]
    for start, end, label in events:
        fig1.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.08,
                       layer="below", line_width=0,
                       annotation_text=label, annotation_position="top left",
                       annotation=dict(font_size=10, font_color=GREY))
    fig1.add_hline(y=0, line_color=GREY, line_width=1)
    fig1.update_layout(
        title="UAE Headline Inflation, 2014–2025 — four macro eras",
        xaxis_title="Date", yaxis_title="YoY (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=500, margin=dict(l=60, r=40, t=60, b=50),
    )
    fig1.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig1.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)
    figures.append(("headline", "Headline CPI", fig1))

    # --- FIG 2: Sector contributions stacked ---
    priority = ["Housing/Utilities", "Transportation", "Food & Beverages",
                "Restaurants/Hotels", "Recreation/Culture", "Miscellaneous"]
    contrib_plot = contrib[contrib.index >= "2021-01-01"].copy()
    colors = [NAVY, TEAL, ACCENT, RED, GREEN, "#8E44AD"]
    fig2 = go.Figure()
    for sec, col in zip(priority, colors):
        if sec in contrib_plot.columns:
            fig2.add_trace(go.Bar(
                x=contrib_plot.index, y=contrib_plot[sec], name=sec,
                marker_color=col,
                hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}} pp<extra></extra>",
            ))
    fig2.add_trace(go.Scatter(
        x=contrib_plot.index, y=contrib_plot["__headline"],
        mode="lines", name="Headline YoY (total)",
        line=dict(color="black", width=2.5),
        hovertemplate="<b>Headline</b><br>%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    fig2.update_layout(
        title="Composition of UAE inflation: Housing is the pivot, Transportation the amplifier",
        barmode="relative", xaxis_title="Date", yaxis_title="Contribution to YoY (pp)",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=550, margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(orientation="h", y=-0.2),
    )
    fig2.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig2.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)
    figures.append(("composition", "Sector Composition", fig2))

    # --- FIG 3: Regime map (UMAP) ---
    fig3 = go.Figure()
    data = coords.copy()
    data["regime"] = labels
    data["date"] = data.index
    data["yoy"] = yoy.loc[data.index, "All Items"]

    # Temporal path
    fig3.add_trace(go.Scatter(
        x=data["umap_1"], y=data["umap_2"],
        mode="lines", line=dict(color=GREY, width=0.7, dash="dot"),
        hoverinfo="skip", showlegend=False, name="Time path",
    ))
    import plotly.express as px
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    for i, lab in enumerate(sorted(set(labels) - {-1})):
        sub = data[data["regime"] == lab]
        start = sub.index.min().strftime("%b %Y")
        end = sub.index.max().strftime("%b %Y")
        fig3.add_trace(go.Scatter(
            x=sub["umap_1"], y=sub["umap_2"],
            mode="markers",
            marker=dict(size=11, color=palette[i % len(palette)],
                        line=dict(color="white", width=1)),
            name=f"Regime {lab}: {start} – {end}",
            customdata=np.column_stack([sub.index.strftime("%b %Y"), sub["yoy"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Regime %{text}<br>Headline: %{customdata[1]:.2f}%<extra></extra>",
            text=[str(lab)] * len(sub),
        ))
    noise = data[data["regime"] == -1]
    if not noise.empty:
        fig3.add_trace(go.Scatter(
            x=noise["umap_1"], y=noise["umap_2"],
            mode="markers",
            marker=dict(size=7, color=GREY, symbol="x"),
            name=f"Transitional ({len(noise)} mo)",
            customdata=np.column_stack([noise.index.strftime("%b %Y"), noise["yoy"]]),
            hovertemplate="<b>%{customdata[0]}</b> (transitional)<br>Headline: %{customdata[1]:.2f}%<extra></extra>",
        ))
    # Current marker
    latest = data.index.max()
    fig3.add_trace(go.Scatter(
        x=[data.loc[latest, "umap_1"]], y=[data.loc[latest, "umap_2"]],
        mode="markers+text",
        marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=RED, width=3)),
        text=[f"  Current: {latest.strftime('%b %Y')}"],
        textposition="middle right", textfont=dict(color=RED, size=12),
        name="Current state", showlegend=False,
    ))
    fig3.update_layout(
        title="Unsupervised regime discovery — HDBSCAN clustering on 13-sector profiles",
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="closest", height=600, margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(orientation="v", x=1.02, y=0.5),
    )
    fig3.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig3.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)
    figures.append(("regimes", "Regime Map", fig3))

    # --- FIG 4: Forecast vs actual ---
    fig4 = go.Figure()
    actuals = ridge_bt.actuals
    fig4.add_trace(go.Scatter(
        x=actuals.index, y=actuals.values,
        mode="lines", name="Actual headline YoY",
        line=dict(color=NAVY, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:.2f}%<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=ridge_bt.predictions.index, y=ridge_bt.predictions.values,
        mode="lines", name="Ridge forecast (3mo ahead)",
        line=dict(color=ACCENT, width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:.2f}%<extra></extra>",
    ))
    fig4.add_trace(go.Scatter(
        x=naive_bt.predictions.index, y=naive_bt.predictions.values,
        mode="lines", name="Naive seasonal baseline",
        line=dict(color=GREY, width=1.3, dash="dash"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Naive: %{y:.2f}%<extra></extra>",
    ))
    fig4.add_hline(y=0, line_color=GREY, line_width=0.5)
    fig4.update_layout(
        title=f"Ridge 3-month forecast vs. actual — RMSE {ridge_bt.rmse:.2f}pp, directional accuracy {ridge_bt.directional_accuracy:.0%}",
        xaxis_title="Date", yaxis_title="YoY (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=500, margin=dict(l=60, r=40, t=60, b=50),
    )
    fig4.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig4.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)
    figures.append(("forecast", "Forecast Timeline", fig4))

    # --- FIG 5: Model benchmark ---
    fig5 = make_subplots(rows=1, cols=2, subplot_titles=("RMSE by horizon (lower = better)", "Directional accuracy (higher = better)"))
    for i, col in enumerate(["naive_seasonal", "ridge", "xgboost"]):
        color = {NAVY: NAVY, "ridge": NAVY, "xgboost": ACCENT, "naive_seasonal": GREY}[col]
        sub = bench[bench["model"] == col]
        fig5.add_trace(go.Bar(
            x=[f"{h}-month" for h in sub["horizon"]], y=sub["rmse"],
            name=col.replace("_", " ").title(), marker_color=color,
            legendgroup=col,
        ), row=1, col=1)
        fig5.add_trace(go.Bar(
            x=[f"{h}-month" for h in sub["horizon"]], y=sub["directional_accuracy"] * 100,
            name=col.replace("_", " ").title(), marker_color=color,
            legendgroup=col, showlegend=False,
        ), row=1, col=2)
    fig5.add_hline(y=50, line_color=RED, line_width=1, line_dash="dash",
                   annotation_text="Coin flip", annotation_position="bottom right", row=1, col=2)
    fig5.update_yaxes(title_text="RMSE (pp)", row=1, col=1, gridcolor=LIGHT_GREY)
    fig5.update_yaxes(title_text="Directional accuracy (%)", row=1, col=2, gridcolor=LIGHT_GREY)
    fig5.update_xaxes(gridcolor=LIGHT_GREY)
    fig5.update_layout(
        title="Model benchmark — walk-forward backtest, 71 origins",
        plot_bgcolor="white", paper_bgcolor="white",
        height=500, margin=dict(l=60, r=40, t=80, b=50),
        barmode="group",
    )
    figures.append(("benchmark", "Model Benchmark", fig5))

    # ---- Assemble standalone HTML ----
    from plotly.io import to_html

    # Build tab navigation
    tab_nav = "".join(
        f'<a href="#{slug}" class="tab">{name}</a>'
        for slug, name, _ in figures
    )

    chart_blocks = ""
    for slug, name, fig in figures:
        fig_html = to_html(fig, include_plotlyjs=False, full_html=False, div_id=f"chart-{slug}")
        chart_blocks += f'<section id="{slug}"><h2>{name}</h2>{fig_html}</section>\n'

    # Custom HTML shell with BCG-style polish
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UAE Inflation Early-Warning System</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    background: #FAFAF7; color: #1F2937;
    margin: 0; padding: 0; line-height: 1.6;
  }}
  header {{
    background: {NAVY}; color: white;
    padding: 2.2rem 2rem 1.5rem;
  }}
  header h1 {{ margin: 0 0 0.4rem; font-size: 1.7rem; font-weight: 600; }}
  header .sub {{ opacity: 0.85; font-size: 0.95rem; }}
  header .tagline {{ color: {ACCENT}; font-weight: 500; margin-top: 0.5rem; }}
  nav {{
    background: white; border-bottom: 1px solid {LIGHT_GREY};
    padding: 0.8rem 2rem; position: sticky; top: 0; z-index: 100;
  }}
  nav .tab {{
    display: inline-block; padding: 0.5rem 1rem; margin-right: 0.3rem;
    color: #4B5563; text-decoration: none; border-radius: 6px;
    font-size: 0.9rem; font-weight: 500;
  }}
  nav .tab:hover {{ background: {LIGHT_GREY}; color: {NAVY}; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }}
  section {{
    background: white; border-radius: 8px;
    padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    scroll-margin-top: 70px;
  }}
  section h2 {{
    margin: 0 0 0.8rem; color: {NAVY}; font-size: 1.15rem; font-weight: 600;
    border-bottom: 2px solid {LIGHT_GREY}; padding-bottom: 0.5rem;
  }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin: 1rem 0 1.5rem;
  }}
  .kpi {{
    background: white; border-left: 4px solid {NAVY}; padding: 1rem 1.2rem;
    border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; }}
  .kpi .label {{ font-size: 0.82rem; color: {GREY}; text-transform: uppercase; letter-spacing: 0.04em; }}
  footer {{
    text-align: center; padding: 2rem; font-size: 0.85rem; color: {GREY};
    border-top: 1px solid {LIGHT_GREY}; margin-top: 3rem;
  }}
  footer a {{ color: {NAVY}; text-decoration: none; }}
  footer a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>

<header>
  <h1>UAE Inflation Early-Warning System</h1>
  <div class="sub">Sector-based forecasting of headline CPI, 2014 – Dec 2025</div>
  <div class="tagline">Ridge regression · 90% directional accuracy at 3 months · walk-forward backtested</div>
</header>

<nav>{tab_nav}</nav>

<main>

<div class="kpi-grid">
  <div class="kpi">
    <div class="value">1.16 pp</div>
    <div class="label">RMSE, 3-month forecast</div>
  </div>
  <div class="kpi">
    <div class="value">90%</div>
    <div class="label">Directional accuracy</div>
  </div>
  <div class="kpi">
    <div class="value">0.75</div>
    <div class="label">Out-of-sample R²</div>
  </div>
  <div class="kpi">
    <div class="value">71</div>
    <div class="label">Walk-forward origins</div>
  </div>
</div>

{chart_blocks}

</main>

<footer>
  <strong>Akanksha Swarnim</strong> · MSc Data Science, University of Birmingham Dubai · 2026<br>
  <a href="https://github.com/AkankshaSwarnim/UAE-CPI-Early-Warning">Source code on GitHub</a> ·
  Data: <a href="https://uaestat.fcsc.gov.ae">FCSA Open Data</a>
</footer>

</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {out_path} ({len(html):,} bytes)")
    return out_path


if __name__ == "__main__":
    build_dashboard("docs/dashboard.html")
