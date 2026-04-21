"""
UAE CPI analytical toolbox.

Implements:
  * STL decomposition (trend/seasonal/residual variance share)
  * CUSUM structural-break detection
  * Granger causality tests (sector YoY -> headline YoY)
  * HDBSCAN regime clustering on a sector z-score matrix
  * UMAP 2-D projection of the regime space
  * Contribution analysis (sector YoY * basket weight)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def stl_variance_share(series: pd.Series, period: int = 12) -> dict:
    """Return share of variance in trend, seasonal, residual components."""
    s = series.dropna()
    stl = STL(s, period=period, robust=True).fit()
    total_var = np.var(s.values)
    return {
        "trend_share": float(np.var(stl.trend.values) / total_var),
        "seasonal_share": float(np.var(stl.seasonal.values) / total_var),
        "residual_share": float(np.var(stl.resid.values) / total_var),
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid,
    }


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

def cusum(series: pd.Series, ref_window: int = 24) -> pd.Series:
    """
    Cumulative sum of deviations from a rolling reference mean.

    Structural breaks show up as inflection points in the CUSUM path.
    """
    s = series.dropna()
    ref = s.rolling(ref_window, min_periods=ref_window).mean()
    dev = s - ref
    return dev.cumsum()


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def granger_sector_to_headline(
    yoy_panel: pd.DataFrame,
    target: str = "All Items",
    max_lag: int = 6,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether each sector's YoY Granger-causes headline YoY.

    For each sector, returns the minimum p-value across lags 1..max_lag and
    the lag at which that minimum occurs. Drops the target column itself.

    NB: Granger causality is statistical prediction, not structural causation.
    Reported as 'predictive lead' in the writeup.
    """
    headline = yoy_panel[target].dropna()
    out = []
    for sector in yoy_panel.columns:
        if sector == target:
            continue
        s = yoy_panel[sector].dropna()
        common = headline.index.intersection(s.index)
        if len(common) < max_lag + 20:
            continue
        data = pd.concat([headline.loc[common], s.loc[common]], axis=1)
        data.columns = ["y", "x"]
        try:
            res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}
            best_lag = min(pvals, key=pvals.get)
            out.append({
                "sector": sector,
                "min_p": pvals[best_lag],
                "best_lag_months": best_lag,
                "significant": pvals[best_lag] < alpha,
            })
        except Exception as e:
            out.append({"sector": sector, "min_p": np.nan, "best_lag_months": np.nan, "significant": False})
    return pd.DataFrame(out).sort_values("min_p").reset_index(drop=True)


def stationarity_check(series: pd.Series) -> dict:
    """ADF test. Low p-value => stationary."""
    s = series.dropna()
    stat, p, *_ = adfuller(s, autolag="AIC")
    return {"adf_stat": float(stat), "p_value": float(p), "stationary": p < 0.05}


# ---------------------------------------------------------------------------
# Regime clustering
# ---------------------------------------------------------------------------

def regime_features(yoy_panel: pd.DataFrame) -> pd.DataFrame:
    """Z-score each sector's YoY across time. Used as the regime feature matrix."""
    X = yoy_panel.dropna()
    scaler = StandardScaler()
    Z = scaler.fit_transform(X.values)
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def hdbscan_regimes(feature_matrix: pd.DataFrame, min_cluster_size: int = 8) -> pd.Series:
    """Fit HDBSCAN on the z-scored sector matrix. Returns cluster labels indexed by date."""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
    labels = clusterer.fit_predict(feature_matrix.values)
    return pd.Series(labels, index=feature_matrix.index, name="regime")


def umap_project(feature_matrix: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """2-D UMAP projection of the regime feature space."""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=10, min_dist=0.1)
    coords = reducer.fit_transform(feature_matrix.values)
    return pd.DataFrame(coords, index=feature_matrix.index, columns=["umap_1", "umap_2"])


# ---------------------------------------------------------------------------
# Contribution analysis
# ---------------------------------------------------------------------------

def contribution_to_headline(
    yoy_panel: pd.DataFrame,
    weights: dict[str, float],
    target: str = "All Items",
) -> pd.DataFrame:
    """
    Approximate contribution (in pp) of each sector to headline YoY.

    contribution_i,t = weight_i * yoy_i,t / 100

    Weights must be in percentage units (e.g. 34.1 for Housing). Contributions
    sum (approximately) to headline YoY, with residual from non-linearity and
    basket-weight rounding.
    """
    yoy = yoy_panel.drop(columns=[target], errors="ignore")
    common = [c for c in yoy.columns if c in weights]
    contrib = yoy[common].copy()
    for c in common:
        contrib[c] = yoy[c] * weights[c] / 100.0
    contrib["__sum"] = contrib.sum(axis=1)
    contrib["__headline"] = yoy_panel[target]
    contrib["__residual"] = contrib["__headline"] - contrib["__sum"]
    return contrib


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_raw, build_sector_panel, yoy_panel as to_yoy, CPI_WEIGHTS

    df = load_raw()
    panel = build_sector_panel(df, start="2014-01-01")
    yoy = to_yoy(panel)

    # STL
    print("STL variance decomposition of headline CPI YoY:")
    dec = stl_variance_share(yoy["All Items"].dropna())
    print(f"  Trend:    {dec['trend_share']:.1%}")
    print(f"  Seasonal: {dec['seasonal_share']:.1%}")
    print(f"  Residual: {dec['residual_share']:.1%}")

    # Granger
    print("\nGranger causality: which sectors lead headline CPI?")
    g = granger_sector_to_headline(yoy, max_lag=6)
    print(g.to_string(index=False))

    # HDBSCAN
    print("\nHDBSCAN regime clustering:")
    feats = regime_features(yoy)
    labels = hdbscan_regimes(feats, min_cluster_size=8)
    print(labels.value_counts().sort_index())
    for lab in sorted(labels.unique()):
        dates = labels[labels == lab].index
        if len(dates) > 0:
            print(f"  Regime {lab}: {dates.min().date()} to {dates.max().date()} ({len(dates)} months)")
