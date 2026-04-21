"""
UAE CPI forecasting engine.

Produces h-month-ahead forecasts of headline YoY inflation using three models
trained on sector-level features:

  1. Naive-seasonal       — YoY_{t+h} = YoY_{t-12+h}  (honest baseline)
  2. Ridge regression     — linear model on lagged sector YoYs
  3. Gradient boosting    — XGBoost on the same feature set

All models are evaluated with expanding-window (walk-forward) backtesting, which
simulates real-world deployment: at each forecast origin, only data up to that
date is used for training. This prevents look-ahead bias.

Metrics reported: RMSE, MAE, directional accuracy, and hit rate vs. the naive
baseline (share of origins where the model beats naive on squared error).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_matrix(
    yoy_panel: pd.DataFrame,
    target_col: str = "All Items",
    sector_cols: list[str] | None = None,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    horizon: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for h-step-ahead forecasting of `target_col` YoY.

    Features are lagged values of each sector's YoY at the specified lags.
    Target is YoY_{t+horizon}.

    Rows with any NaN (start-of-sample or end-of-sample) are dropped.
    """
    if sector_cols is None:
        sector_cols = [c for c in yoy_panel.columns if c != target_col]

    feats = {}
    for col in [target_col] + sector_cols:
        for lag in lags:
            feats[f"{col}__lag{lag}"] = yoy_panel[col].shift(lag)
    X = pd.DataFrame(feats, index=yoy_panel.index)

    # Target is YoY h months ahead
    y = yoy_panel[target_col].shift(-horizon)
    y.name = f"{target_col}__t+{horizon}"

    # Align and drop NaNs
    aligned = pd.concat([X, y], axis=1).dropna()
    X_clean = aligned.drop(columns=[y.name])
    y_clean = aligned[y.name]
    return X_clean, y_clean


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_ridge(alpha: float = 1.0):
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=0))])


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=0,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    predictions: pd.Series
    actuals: pd.Series
    rmse: float
    mae: float
    directional_accuracy: float  # share of correctly signed changes vs. origin
    r2: float

    def summary(self) -> dict:
        return {
            "model": self.name,
            "rmse": round(self.rmse, 3),
            "mae": round(self.mae, 3),
            "directional_accuracy": round(self.directional_accuracy, 3),
            "r2": round(self.r2, 3),
            "n": len(self.actuals),
        }


def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn: Callable,
    name: str,
    min_train: int = 48,
    origin_yoy: pd.Series | None = None,
) -> BacktestResult:
    """
    Expanding-window backtest.

    At each forecast origin t (with at least `min_train` prior observations),
    refit the model on data up to t and predict y[t]. Returns predictions
    aligned with actuals.

    `origin_yoy` is the current headline YoY at each origin — needed to
    compute directional accuracy (did YoY rise or fall from origin to target?).
    """
    preds = []
    idx = []
    for i in range(min_train, len(X)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]]
        model = model_fn()
        model.fit(X_train, y_train)
        yhat = float(model.predict(X_test)[0])
        preds.append(yhat)
        idx.append(X.index[i])

    predictions = pd.Series(preds, index=idx, name=f"{name}_pred")
    actuals = y.loc[predictions.index]

    err = predictions - actuals
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((actuals - actuals.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Directional accuracy: did the model predict the correct sign of change?
    if origin_yoy is not None:
        common = predictions.index.intersection(origin_yoy.index)
        pred_dir = np.sign(predictions.loc[common] - origin_yoy.loc[common])
        actual_dir = np.sign(actuals.loc[common] - origin_yoy.loc[common])
        da = float((pred_dir == actual_dir).mean())
    else:
        da = np.nan

    return BacktestResult(name, predictions, actuals, rmse, mae, da, r2)


def naive_seasonal_backtest(
    yoy_series: pd.Series,
    target_index: pd.Index,
    horizon: int = 3,
) -> BacktestResult:
    """
    Naive baseline: forecast YoY_{t+h} = YoY_{t-12+h}.

    This isn't a random-walk baseline — it's the strongest naive benchmark
    because it at least captures the seasonal structure. Beating this is
    meaningful.
    """
    preds = {}
    actuals = {}
    origins = {}
    for dt in target_index:
        # Target date
        origin = dt - pd.DateOffset(months=horizon)
        ref = dt - pd.DateOffset(months=12)
        if origin in yoy_series.index and ref in yoy_series.index and dt in yoy_series.index:
            preds[dt] = yoy_series.loc[ref]
            actuals[dt] = yoy_series.loc[dt]
            origins[dt] = yoy_series.loc[origin]

    predictions = pd.Series(preds, name="naive_pred")
    actuals_s = pd.Series(actuals)
    origins_s = pd.Series(origins)

    err = predictions - actuals_s
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    ss_res = float((err ** 2).sum())
    ss_tot = float(((actuals_s - actuals_s.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pred_dir = np.sign(predictions - origins_s)
    actual_dir = np.sign(actuals_s - origins_s)
    da = float((pred_dir == actual_dir).mean())

    return BacktestResult("naive_seasonal", predictions, actuals_s, rmse, mae, da, r2)


# ---------------------------------------------------------------------------
# Convenience: full benchmark
# ---------------------------------------------------------------------------

def run_full_benchmark(
    yoy_panel: pd.DataFrame,
    target_col: str = "All Items",
    horizons: tuple[int, ...] = (1, 3, 6),
    min_train: int = 48,
) -> pd.DataFrame:
    """Run naive / ridge / xgb backtest at multiple horizons. Returns a tidy frame."""
    rows = []
    for h in horizons:
        X, y = build_feature_matrix(yoy_panel, target_col=target_col, horizon=h)
        yoy_target = yoy_panel[target_col]

        naive = naive_seasonal_backtest(yoy_target, y.index[min_train:], horizon=h)
        rows.append({"horizon": h, **naive.summary()})

        for name, fn in [("ridge", make_ridge), ("xgboost", make_xgb)]:
            origin = yoy_target.shift(h)
            bt = walk_forward_backtest(X, y, fn, name, min_train=min_train, origin_yoy=origin)
            rows.append({"horizon": h, **bt.summary()})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_raw, build_sector_panel, yoy_panel as to_yoy

    df = load_raw()
    panel = build_sector_panel(df, start="2014-01-01")
    yoy = to_yoy(panel)

    print("Running full benchmark (horizons: 1, 3, 6 months)...")
    results = run_full_benchmark(yoy, horizons=(1, 3, 6))
    print("\n" + results.to_string(index=False))
