"""Churn model with walk-forward validation.

We define churn as 'no purchase in the 90 days after a cutoff date'.
For each cutoff:
  - Features come from data BEFORE the cutoff
  - Label comes from the 90-day window AFTER the cutoff
We never train on data from the future of the validation cutoff.

Walk-forward folds:
  Fold 1: train @ 2010-06-30 -> validate @ 2010-12-31
  Fold 2: train @ 2010-12-31 -> validate @ 2011-06-30
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
DOC_OUT = Path("docs/churn_model.md")
CSV_OUT = Path("data/staging/churn_scores.csv")

CUTOFFS = ["2010-06-30", "2010-12-31", "2011-06-30"]
CHURN_WINDOW_DAYS = 90


def build_features(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    """Compute one row per customer using ONLY data strictly before cutoff."""
    q = f"""
    WITH filtered AS (
        SELECT
            CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS customer_id,
            invoice_no,
            invoice_date,
            revenue,
            stock_code
        FROM fact_sales
        WHERE NOT is_cancellation
          AND NOT is_guest
          AND NOT is_non_product
          AND NOT is_invalid_price
          AND customer_id IS NOT NULL
          AND invoice_date < DATE '{cutoff}'
    )
    SELECT
        customer_id,
        MAX(invoice_date) AS last_purchase,
        MIN(invoice_date) AS first_purchase,
        COUNT(DISTINCT invoice_no) AS frequency,
        SUM(revenue) AS monetary,
        COUNT(DISTINCT stock_code) AS product_diversity,
        COUNT(DISTINCT invoice_date::DATE) AS active_days
    FROM filtered
    GROUP BY customer_id
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    cutoff_ts = pd.Timestamp(cutoff)
    df["recency_days"] = (cutoff_ts - pd.to_datetime(df["last_purchase"])).dt.days
    df["tenure_days"] = (cutoff_ts - pd.to_datetime(df["first_purchase"])).dt.days
    df["avg_order_value"] = df["monetary"] / df["frequency"].clip(lower=1)
    df["purchase_cadence"] = df["tenure_days"] / df["frequency"].clip(lower=1)
    return df


def build_labels(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    """Label = churned (no purchase in 90 days after cutoff)."""
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=CHURN_WINDOW_DAYS)
    q = f"""
    SELECT DISTINCT
        CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS customer_id
    FROM fact_sales
    WHERE NOT is_cancellation
      AND NOT is_guest
      AND NOT is_non_product
      AND NOT is_invalid_price
      AND customer_id IS NOT NULL
      AND invoice_date >= DATE '{start.date()}'
      AND invoice_date <  DATE '{end.date()}'
    """
    active = con.execute(q).fetchdf()
    return active


def build_dataset(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    features = build_features(con, cutoff)
    if features.empty:
        return features
    active_after = build_labels(con, cutoff)
    features["churned"] = (~features["customer_id"].isin(active_after["customer_id"])).astype(int)
    return features


FEATURE_COLS = [
    "recency_days", "tenure_days", "frequency", "monetary",
    "product_diversity", "avg_order_value", "purchase_cadence",
    "active_days",
]


def prep_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, df["churned"]


def evaluate(y_true: pd.Series, y_prob: np.ndarray, name: str) -> dict:
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    # Precision at top 10% (highest risk)
    threshold = np.quantile(y_prob, 0.90)
    pred_top10 = (y_prob >= threshold).astype(int)
    prec10 = precision_score(y_true, pred_top10, zero_division=0)
    lift10 = prec10 / max(y_true.mean(), 1e-6)
    return {
        "model": name,
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "precision_top10pct": round(prec10, 4),
        "lift_top10pct": round(lift10, 2),
        "base_rate": round(float(y_true.mean()), 4),
        "n": len(y_true),
    }


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("=" * 78)
    print("BUILDING DATASETS FOR EACH CUTOFF")
    print("=" * 78)
    datasets: dict[str, pd.DataFrame] = {}
    for cutoff in CUTOFFS:
        df = build_dataset(con, cutoff)
        datasets[cutoff] = df
        if df.empty:
            print(f"  {cutoff}: EMPTY")
        else:
            churn_rate = df["churned"].mean()
            print(f"  {cutoff}: {len(df):,} customers   churn rate {churn_rate*100:.1f}%")

    # Walk-forward folds
    folds = [
        ("2010-06-30", "2010-12-31"),
        ("2010-12-31", "2011-06-30"),
    ]

    results = []
    final_scores: pd.DataFrame | None = None

    print()
    print("=" * 78)
    print("WALK-FORWARD EVALUATION")
    print("=" * 78)

    for train_cutoff, val_cutoff in folds:
        train_df = datasets[train_cutoff]
        val_df = datasets[val_cutoff]
        if train_df.empty or val_df.empty:
            print(f"  Fold {train_cutoff} -> {val_cutoff}: SKIP (empty)")
            continue

        X_train, y_train = prep_xy(train_df)
        X_val, y_val = prep_xy(val_df)

        # Logistic baseline (scaled)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        logit = LogisticRegression(max_iter=1000, class_weight="balanced")
        logit.fit(X_train_s, y_train)
        prob_lr = logit.predict_proba(X_val_s)[:, 1]

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb.set_params(scale_pos_weight=scale_pos)
        xgb.fit(X_train, y_train)
        prob_xgb = xgb.predict_proba(X_val)[:, 1]

        lr_metrics = evaluate(y_val, prob_lr, "Logistic")
        xgb_metrics = evaluate(y_val, prob_xgb, "XGBoost")
        lr_metrics["fold"] = f"{train_cutoff}->{val_cutoff}"
        xgb_metrics["fold"] = f"{train_cutoff}->{val_cutoff}"

        results.extend([lr_metrics, xgb_metrics])

        print(f"\n  Fold: train on {train_cutoff} -> validate on {val_cutoff}")
        print(f"    {'Model':<12}{'ROC-AUC':>10}{'PR-AUC':>10}{'Prec@10%':>12}{'Lift@10%':>10}{'Base':>8}")
        for m in [lr_metrics, xgb_metrics]:
            print(f"    {m['model']:<12}{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}"
                  f"{m['precision_top10pct']:>12.3f}{m['lift_top10pct']:>10.2f}{m['base_rate']:>8.2f}")

        # Save scores from the latest fold
        if val_cutoff == "2011-06-30":
            final_scores = val_df[["customer_id", "recency_days", "frequency", "monetary"]].copy()
            final_scores["churn_probability"] = prob_xgb
            final_scores["churn_risk"] = pd.cut(
                final_scores["churn_probability"],
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            ).astype(str)

    # Summary table
    results_df = pd.DataFrame(results)
    print()
    print("=" * 78)
    print("SUMMARY: AVERAGE ACROSS FOLDS")
    print("=" * 78)
    if not results_df.empty:
        summary = results_df.groupby("model")[["roc_auc", "pr_auc", "precision_top10pct", "lift_top10pct"]].mean().round(3)
        print(summary.to_string())

    # SHAP on latest XGBoost
    print()
    print("=" * 78)
    print("FEATURE IMPORTANCE (XGBoost, latest fold)")
    print("=" * 78)
    try:
        import shap
        train_df = datasets["2010-12-31"]
        X_train, y_train = prep_xy(train_df)
        xgb_final = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False, random_state=42,
        )
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_final.set_params(scale_pos_weight=scale_pos)
        xgb_final.fit(X_train, y_train)

        explainer = shap.TreeExplainer(xgb_final)
        shap_values = explainer.shap_values(X_train)
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<22} {row['mean_abs_shap']:.4f}")
    except Exception as e:
        print(f"  SHAP skipped: {e}")
        importance = pd.DataFrame()

    # Save final scores
    if final_scores is not None and not final_scores.empty:
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        final_scores.to_csv(CSV_OUT, index=False)
        con.execute("DROP TABLE IF EXISTS churn_scores")
        con.register("cs", final_scores)
        con.execute("CREATE TABLE churn_scores AS SELECT * FROM cs")
        con.unregister("cs")
        print(f"\nSaved {len(final_scores):,} scored customers -> {CSV_OUT}")
        print("Saved table churn_scores to warehouse")

    # Model card
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Churn Model Card", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Problem", "",
        f"Predict whether a customer will churn (no purchase in {CHURN_WINDOW_DAYS} days after cutoff).", "",
        "## Walk-forward validation", "",
        "| Fold | Train cutoff | Validate cutoff |",
        "|---|---|---|",
    ]
    for tr, va in folds:
        lines.append(f"| {tr} -> {va} | {tr} | {va} |")
    lines += ["", "## Metrics per fold", "", "| Fold | Model | ROC-AUC | PR-AUC | Prec@10% | Lift@10% |",
              "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['fold']} | {r['model']} | {r['roc_auc']} | {r['pr_auc']} | {r['precision_top10pct']} | {r['lift_top10pct']} |")
    lines += ["", "## Features", ""] + [f"- {c}" for c in FEATURE_COLS]
    if not importance.empty:
        lines += ["", "## Feature importance (SHAP mean abs)", "",
                  "| Feature | Mean abs SHAP |", "|---|---|"]
        for _, row in importance.iterrows():
            lines.append(f"| {row['feature']} | {row['mean_abs_shap']:.4f} |")
    lines += ["", "## Limitations", "",
              "- Dataset has no marketing/promo data; model can't account for campaigns.",
              "- Q4 seasonality may inflate churn labels in Q1 cutoffs.",
              "- Guest customers excluded from modeling.",
              "- Model predicts 90-day churn on this retailer; may need re-tuning for other domains."]
    DOC_OUT.write_text("\n".join(lines) + "\n")
    print(f"Saved: {DOC_OUT}")

    con.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
