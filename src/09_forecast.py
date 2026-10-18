"""Monthly revenue forecasting with time-series cross-validation.

Compares:
  - Seasonal naive (baseline)
  - SARIMA
  - Holt-Winters exponential smoothing
  - LightGBM with lag features

Uses rolling-origin time-series CV (expanding window),
never random train/test split.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
CSV_OUT = Path("data/staging/revenue_forecast.csv")
DOC_OUT = Path("docs/forecast.md")

HORIZON = 6  # forecast 6 months ahead


def load_monthly_revenue(con: duckdb.DuckDBPyConnection) -> pd.Series:
    q = """
    SELECT
        DATE_TRUNC('month', invoice_date)::DATE AS month,
        SUM(revenue) AS revenue
    FROM fact_sales
    WHERE NOT is_cancellation
      AND NOT is_invalid_price
      AND NOT is_non_product
    GROUP BY month
    ORDER BY month
    """
    df = con.execute(q).fetchdf()
    s = pd.Series(df["revenue"].values, index=pd.to_datetime(df["month"]), name="revenue")
    # Fill missing months with 0
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full_idx).fillna(0)
    return s


def seasonal_naive(history: pd.Series, horizon: int, seasonality: int = 12) -> np.ndarray:
    """Forecast = same month last year."""
    last_year = history.iloc[-seasonality:].values
    if len(last_year) < horizon:
        return np.resize(last_year, horizon)
    return last_year[-horizon:]


def fit_sarima(history: pd.Series, horizon: int) -> np.ndarray:
    try:
        model = SARIMAX(
            history,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        return fit.forecast(horizon).values
    except Exception:
        return np.full(horizon, np.nan)


def fit_holt_winters(history: pd.Series, horizon: int) -> np.ndarray:
    try:
        # Need at least 2 seasonal periods
        seasonal_periods = 12 if len(history) >= 24 else None
        if seasonal_periods is None:
            model = ExponentialSmoothing(history, trend="add")
        else:
            model = ExponentialSmoothing(
                history, trend="add", seasonal="add", seasonal_periods=seasonal_periods
            )
        fit = model.fit(optimized=True, use_brute=False)
        return fit.forecast(horizon).values
    except Exception:
        return np.full(horizon, np.nan)


def fit_lightgbm(history: pd.Series, horizon: int) -> np.ndarray:
    if not HAS_LGBM or len(history) < 18:
        return np.full(horizon, np.nan)
    try:
        df = pd.DataFrame({"y": history.values}, index=history.index)
        df["month"] = df.index.month
        df["year"] = df.index.year
        df["trend"] = np.arange(len(df))
        for lag in [1, 2, 3, 6, 12]:
            df[f"lag{lag}"] = df["y"].shift(lag)
        df = df.dropna()

        feats = [c for c in df.columns if c != "y"]
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=8, verbose=-1,
        )
        model.fit(df[feats], df["y"])

        # Recursive forecast
        preds = []
        last_df = df.iloc[-1:].copy()
        for h in range(horizon):
            next_idx = history.index[-1] + pd.DateOffset(months=h + 1)
            row = last_df.copy()
            row.index = [next_idx]
            row["month"] = next_idx.month
            row["year"] = next_idx.year
            row["trend"] = len(df) + h
            for lag in [1, 2, 3, 6, 12]:
                idx = -lag + h
                row[f"lag{lag}"] = history.iloc[idx] if -idx <= len(history) else np.nan
            row = row[feats].fillna(0)
            pred = model.predict(row)[0]
            preds.append(pred)
            last_df = row
        return np.array(preds)
    except Exception:
        return np.full(horizon, np.nan)


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Loading monthly revenue ...")
    monthly = load_monthly_revenue(con)
    print(f"  months: {len(monthly)}")
    print(f"  range: {monthly.index.min().date()} -> {monthly.index.max().date()}")
    print(f"  total: GBP {monthly.sum():,.2f}")

    # Rolling-origin CV: hold out last HORIZON months, train on everything before
    if len(monthly) < HORIZON + 12:
        print("ERROR: not enough months for forecasting (need 18+).")
        con.close()
        return 1

    train = monthly.iloc[:-HORIZON]
    actual = monthly.iloc[-HORIZON:]

    print()
    print("=" * 78)
    print(f"TIME-SERIES CV: train on first {len(train)} months, validate on last {HORIZON}")
    print("=" * 78)

    results = {}

    # Seasonal naive
    pred_naive = seasonal_naive(train, HORIZON)
    results["Seasonal Naive"] = pred_naive

    # SARIMA
    print("  fitting SARIMA ...")
    pred_sarima = fit_sarima(train, HORIZON)
    results["SARIMA"] = pred_sarima

    # Holt-Winters
    print("  fitting Holt-Winters ...")
    pred_hw = fit_holt_winters(train, HORIZON)
    results["Holt-Winters"] = pred_hw

    # LightGBM
    if HAS_LGBM:
        print("  fitting LightGBM ...")
        pred_lgbm = fit_lightgbm(train, HORIZON)
        results["LightGBM"] = pred_lgbm

    actual_arr = actual.values

    print()
    print(f"{'Model':<18}{'MAPE %':>10}{'RMSE':>15}{'Total (GBP)':>18}")
    print("-" * 61)
    metrics = []
    for name, pred in results.items():
        if np.isnan(pred).all():
            print(f"{name:<18}{'failed':>10}")
            continue
        m = mape(actual_arr, pred)
        r = rmse(actual_arr, pred)
        total = pred.sum()
        print(f"{name:<18}{m:>10.2f}{r:>15,.0f}{total:>18,.2f}")
        metrics.append({"model": name, "mape": m, "rmse": r, "total": total})

    print("-" * 61)
    print(f"{'Actual':<18}{'':>10}{'':>15}{actual_arr.sum():>18,.2f}")

    # Best model
    metrics_df = pd.DataFrame(metrics)
    best_model = metrics_df.loc[metrics_df["mape"].idxmin(), "model"] if not metrics_df.empty else None
    print()
    print(f"Best model by MAPE: {best_model}")

    # Full-history forecast with best model
    print()
    print("=" * 78)
    print(f"6-MONTH FORECAST USING BEST MODEL ({best_model})")
    print("=" * 78)
    full_pred = results.get(best_model)
    if full_pred is None:
        full_pred = seasonal_naive(monthly, HORIZON)
    last_month = monthly.index.max()
    forecast_dates = pd.date_range(
        last_month + pd.DateOffset(months=1), periods=HORIZON, freq="MS"
    )
    for d, v in zip(forecast_dates, full_pred):
        print(f"  {d.date()}   GBP {v:>14,.2f}")

    # Save
    forecast_df = pd.DataFrame({"month": forecast_dates, "forecast_revenue": full_pred})
    forecast_df["model"] = best_model
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(CSV_OUT, index=False)
    con.execute("DROP TABLE IF EXISTS revenue_forecast")
    con.register("fdf", forecast_df)
    con.execute("CREATE TABLE revenue_forecast AS SELECT * FROM fdf")
    con.unregister("fdf")

    # Historical for Power BI
    hist_df = pd.DataFrame({"month": monthly.index, "revenue": monthly.values})
    con.execute("DROP TABLE IF EXISTS revenue_monthly")
    con.register("hdf", hist_df)
    con.execute("CREATE TABLE revenue_monthly AS SELECT * FROM hdf")
    con.unregister("hdf")

    # Docs
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Revenue Forecast", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Data range: {monthly.index.min().date()} to {monthly.index.max().date()}",
        f"Validation: last {HORIZON} months held out, trained on {len(train)} months", "",
        "## Model comparison", "",
        "| Model | MAPE % | RMSE | Total forecast (GBP) |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(f"| {m['model']} | {m['mape']:.2f} | {m['rmse']:,.0f} | {m['total']:,.2f} |")
    lines += ["", f"**Best model:** {best_model} (lowest MAPE)", "", "## Next 6 months", "",
              "| Month | Forecast (GBP) |", "|---|---|"]
    for d, v in zip(forecast_dates, full_pred):
        lines.append(f"| {d.date()} | {v:,.2f} |")
    DOC_OUT.write_text("\n".join(lines) + "\n")

    con.close()
    print()
    print(f"Saved: {CSV_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
