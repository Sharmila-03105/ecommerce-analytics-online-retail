"""Survival analysis: time-to-churn using Kaplan-Meier and Cox PH."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
DOC_OUT = Path("docs/survival_analysis.md")
KM_OUT = Path("data/staging/km_survival.csv")
COX_OUT = Path("data/staging/cox_hazard_ratios.csv")


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Building survival dataset ...")

    # Per customer: tenure (days from first to last purchase) + churned flag
    # Churned = hasn't purchased in > 180 days at reference date
    query = """
    WITH c AS (
        SELECT
            CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS customer_id,
            MIN(invoice_date) AS first_purchase,
            MAX(invoice_date) AS last_purchase,
            COUNT(DISTINCT invoice_no) AS frequency,
            SUM(revenue) AS monetary,
            COUNT(DISTINCT stock_code) AS product_diversity,
            COUNT(DISTINCT country) AS country_diversity
        FROM fact_sales
        WHERE NOT is_cancellation
          AND NOT is_guest
          AND NOT is_non_product
          AND NOT is_invalid_price
          AND customer_id IS NOT NULL
        GROUP BY customer_id
        HAVING COUNT(DISTINCT invoice_no) >= 2
    )
    SELECT * FROM c
    """
    df = con.execute(query).fetchdf()
    print(f"  repeat customers: {len(df):,}")

    if df.empty:
        print("ERROR: no data.")
        return 1

    ref_date = pd.to_datetime(df["last_purchase"]).max() + pd.Timedelta("1D")
    print(f"  reference date: {ref_date.date()}")

    df["tenure_days"] = (pd.to_datetime(df["last_purchase"]) -
                          pd.to_datetime(df["first_purchase"])).dt.days
    df["recency_days"] = (ref_date - pd.to_datetime(df["last_purchase"])).dt.days
    df["avg_order_value"] = df["monetary"] / df["frequency"].clip(lower=1)

    # Churn definition: no purchase for 180+ days after last purchase
    df["churned"] = (df["recency_days"] >= 180).astype(int)

    # Join RFM segment if available
    try:
        rfm = con.execute(
            "SELECT customer_id, segment FROM customer_rfm"
        ).fetchdf()
        rfm["customer_id"] = rfm["customer_id"].astype("int64")
        df = df.merge(rfm, on="customer_id", how="left")
        df["segment"] = df["segment"].fillna("Unknown")
    except Exception:
        df["segment"] = "Unknown"

    print(f"  churned: {int(df['churned'].sum()):,} ({df['churned'].mean()*100:.1f}%)")
    print()

    # Kaplan-Meier per RFM segment
    print("=" * 78)
    print("KAPLAN-MEIER SURVIVAL BY SEGMENT")
    print("=" * 78)
    kmf = KaplanMeierFitter()
    km_rows = []
    for seg in df["segment"].unique():
        if seg == "Unknown":
            continue
        seg_df = df[df["segment"] == seg]
        if len(seg_df) < 30:
            continue
        kmf.fit(
            durations=seg_df["tenure_days"].clip(lower=1),
            event_observed=seg_df["churned"],
            label=seg,
        )
        median = kmf.median_survival_time_
        km_rows.append({
            "segment": seg,
            "n": len(seg_df),
            "median_survival_days": median if pd.notna(median) else None,
        })
        print(f"  {seg:<14}  n={len(seg_df):>5,}   median survival: {median:.0f} days" if pd.notna(median)
              else f"  {seg:<14}  n={len(seg_df):>5,}   median survival: >max")

    km_df = pd.DataFrame(km_rows)
    KM_OUT.parent.mkdir(parents=True, exist_ok=True)
    km_df.to_csv(KM_OUT, index=False)

    # Cox PH on numeric features
    print()
    print("=" * 78)
    print("COX PROPORTIONAL HAZARDS")
    print("=" * 78)

    cox_features = ["frequency", "monetary", "avg_order_value",
                    "product_diversity", "country_diversity"]
    cox_df = df[cox_features + ["tenure_days", "churned"]].dropna().copy()
    cox_df["tenure_days"] = cox_df["tenure_days"].clip(lower=1)

    if len(cox_df) < 100:
        print("  Not enough rows for Cox PH")
    else:
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(cox_df, duration_col="tenure_days", event_col="churned")
            summary = cph.summary[["coef", "exp(coef)", "p"]].round(4)
            summary.columns = ["coef", "hazard_ratio", "p_value"]
            summary = summary.sort_values("hazard_ratio", ascending=False)
            print(summary.to_string())
            print()
            print("  Interpretation: hazard_ratio > 1 means feature INCREASES churn risk")
            print("                  hazard_ratio < 1 means feature DECREASES churn risk")
            summary.to_csv(COX_OUT)
            print(f"\n  Concordance index: {cph.concordance_index_:.3f}")
        except Exception as e:
            print(f"  Cox PH failed: {e}")

    # Docs
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Survival Analysis", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Reference date: {ref_date.date()}",
        f"Repeat customers: {len(df):,}",
        f"Churn rate (180-day): {df['churned'].mean()*100:.1f}%", "",
        "## Median survival by RFM segment", "",
        "| Segment | N | Median survival (days) |",
        "|---|---|---|",
    ]
    for row in km_rows:
        median = row["median_survival_days"]
        lines.append(f"| {row['segment']} | {row['n']} | {median if median else '>max'} |")
    DOC_OUT.write_text("\n".join(lines) + "\n")

    # Save survival features for dashboard
    surv_out = df[[
        "customer_id", "tenure_days", "recency_days", "churned",
        "frequency", "monetary", "segment"
    ]].copy()
    surv_out["customer_id"] = surv_out["customer_id"].astype(int)
    con.execute("DROP TABLE IF EXISTS customer_survival")
    con.register("sdf", surv_out)
    con.execute("CREATE TABLE customer_survival AS SELECT * FROM sdf")
    con.unregister("sdf")

    con.close()
    print()
    print(f"Saved: {KM_OUT}")
    if COX_OUT.exists():
        print(f"Saved: {COX_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Saved table customer_survival to warehouse")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
