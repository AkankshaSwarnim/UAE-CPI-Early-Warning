"""Generate all publication figures for the project."""
import sys
from pathlib import Path
sys.path.insert(0, "src")

import pandas as pd
from data_loader import load_raw, build_chained_index, build_sector_panel, yoy_panel as to_yoy, mom_panel, CPI_WEIGHTS
from analysis import (
    stl_variance_share, cusum, granger_sector_to_headline,
    regime_features, hdbscan_regimes, umap_project,
    contribution_to_headline,
)
from forecast import run_full_benchmark, build_feature_matrix, make_ridge, walk_forward_backtest, naive_seasonal_backtest
from plots import (
    plot_headline_cpi, plot_stl, plot_sector_contributions,
    plot_regime_map, plot_forecast_backtest, plot_forecast_timeline, plot_granger_bars
)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

print("Loading data...")
df = load_raw()
chained = build_chained_index(df, "ALL")
chained = chained[chained["Date"] >= "2014-01-01"].reset_index(drop=True)
panel = build_sector_panel(df, start="2014-01-01")
yoy = to_yoy(panel)
mom = mom_panel(panel)

print("1. Headline CPI chart...")
plot_headline_cpi(chained, FIG_DIR / "01_headline_cpi.png")

print("2. STL decomposition...")
stl = stl_variance_share(chained.set_index("Date")["YoY"].dropna())
print(f"   Trend={stl['trend_share']:.1%}, Seasonal={stl['seasonal_share']:.1%}, Residual={stl['residual_share']:.1%}")
plot_stl(chained, stl, FIG_DIR / "02_stl_decomposition.png")

print("3. Sector contributions...")
contrib = contribution_to_headline(yoy, CPI_WEIGHTS)
plot_sector_contributions(contrib, FIG_DIR / "03_sector_contributions.png", start="2021-01-01")

print("4. HDBSCAN regimes (min_cluster_size=6)...")
feats = regime_features(yoy)
labels = hdbscan_regimes(feats, min_cluster_size=6)
print(f"   Found {len(set(labels) - {-1})} regimes + {(labels == -1).sum()} transitional months")
print("5. UMAP projection...")
coords = umap_project(feats)
plot_regime_map(coords, labels, FIG_DIR / "04_regime_map.png")

print("6. Granger causality (MoM)...")
granger = granger_sector_to_headline(mom, max_lag=6)
print(granger.to_string(index=False))
plot_granger_bars(granger, FIG_DIR / "05_granger_causality.png")

print("7. Forecast backtest...")
bt = run_full_benchmark(yoy, horizons=(1, 3, 6))
print(bt.to_string(index=False))
bt.to_csv("outputs/backtest_results.csv", index=False)
plot_forecast_backtest(bt, FIG_DIR / "06_forecast_benchmark.png")

print("8. Forecast timeline at 3-month horizon...")
X, y = build_feature_matrix(yoy, horizon=3)
origin_yoy = yoy["All Items"].shift(3)
ridge_bt = walk_forward_backtest(X, y, make_ridge, "ridge", min_train=48, origin_yoy=origin_yoy)
naive_bt = naive_seasonal_backtest(yoy["All Items"], y.index[48:], horizon=3)
plot_forecast_timeline(ridge_bt.predictions, ridge_bt.actuals, naive_bt.predictions,
                       FIG_DIR / "07_forecast_timeline.png", horizon=3)

# Save key outputs
regime_df = pd.DataFrame({"regime": labels})
regime_df.to_csv("outputs/regimes.csv")
granger.to_csv("outputs/granger_results.csv", index=False)
contrib.to_csv("outputs/contributions.csv")

# Save 6-month forecast for "current outlook"
print("\n9. Generating current 6-month forecast...")
X6, y6 = build_feature_matrix(yoy, horizon=6)
origin6 = yoy["All Items"].shift(6)
ridge6 = walk_forward_backtest(X6, y6, make_ridge, "ridge", min_train=48, origin_yoy=origin6)

# The latest feature row predicts 6 months ahead of the last observation
X_latest = X6.iloc[[-1]]
model = make_ridge()
model.fit(X6.iloc[:-1], y6.iloc[:-1])
forecast_value = float(model.predict(X_latest)[0])
forecast_date = yoy.index[-1] + pd.DateOffset(months=6)
print(f"   Ridge 6-month forecast for {forecast_date.strftime('%b %Y')}: {forecast_value:.2f}% YoY")
print(f"   Current YoY ({yoy.index[-1].strftime('%b %Y')}): {yoy['All Items'].iloc[-1]:.2f}%")

print("\nAll figures saved to", FIG_DIR.absolute())
