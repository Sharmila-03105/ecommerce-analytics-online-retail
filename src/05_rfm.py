"""RFM segmentation: assigns each customer to a business segment."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
CSV_OUT = Path("data/staging/customer_rfm.csv")
DOC_OUT = Path("docs/rfm_summary.md")


def assign_segment(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and f >= 4:
        return "At Risk"
    if r >= 4:
        return "New"
    if r == 1:
        return "Lost"
    return "Hibernating"


def clean_id(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found. Run 03_warehouse.py first.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Pulling customer aggregates ...")
    df = con.execute("""
        SELECT
            customer_id,
            MIN(invoice_date) AS first_purchase,
            MAX(invoice_date) AS last_purchase,
            COUNT(DISTINCT invoice_no) AS frequency,
            SUM(revenue) AS monetary
        FROM fact_sales
        WHERE NOT is_cancellation
          AND NOT is_guest
          AND NOT is_non_product
          AND NOT is_invalid_price
          AND customer_id IS NOT NULL
        GROUP BY customer_id
    """).fetchdf()

    print(f"  customers: {len(df):,}")

    # Clean customer_id (Excel floats -> ints)
    df["customer_id"] = df["customer_id"].apply(clean_id)
    df = df[df["customer_id"].notna()].copy()
    df["customer_id"] = df["customer_id"].astype(int)

    # Recency
    last_purchase = pd.to_datetime(df["last_purchase"])
    ref_date = last_purchase.max() + pd.Timedelta("1D")
    df["recency"] = (ref_date - last_purchase).dt.days
    print(f"  reference date: {ref_date.date()}")

    # FIXED: lowest recency (recent buyer) -> highest R score
    df["R"] = pd.qcut(df["recency"].rank(method="first", ascending=True),
                      5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F"] = pd.qcut(df["frequency"].rank(method="first", ascending=True),
                      5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M"] = pd.qcut(df["monetary"].rank(method="first", ascending=True),
                      5, labels=[1, 2, 3, 4, 5]).astype(int)

    df["rfm_score"] = df["R"].astype(str) + df["F"].astype(str) + df["M"].astype(str)
    df["segment"] = [assign_segment(r, f, m) for r, f, m in zip(df["R"], df["F"], df["M"])]

    con.execute("DROP TABLE IF EXISTS customer_rfm")
    con.register("rfm_df", df)
    con.execute("CREATE TABLE customer_rfm AS SELECT * FROM rfm_df")
    con.unregister("rfm_df")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    total_customers = len(df)
    total_revenue = df["monetary"].sum()

    summary = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency_days=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    summary["customer_pct"] = (summary["customers"] / total_customers * 100).round(1)
    summary["revenue_pct"] = (summary["revenue"] / total_revenue * 100).round(1)

    print()
    print("=" * 78)
    print("SEGMENT SUMMARY")
    print("=" * 78)
    print(f"{'Segment':<14}{'Customers':>10}{'%':>7}{'Revenue (GBP)':>16}{'%':>7}{'Avg Rec':>9}{'Avg Freq':>10}")
    print("-" * 78)
    for _, row in summary.iterrows():
        print(
            f"{row['segment']:<14}"
            f"{row['customers']:>10,}"
            f"{row['customer_pct']:>6.1f}%"
            f"{row['revenue']:>16,.2f}"
            f"{row['revenue_pct']:>6.1f}%"
            f"{row['avg_recency_days']:>9.0f}"
            f"{row['avg_frequency']:>10.1f}"
        )
    print("-" * 78)
    print(f"{'TOTAL':<14}{total_customers:>10,}{'':>7}{total_revenue:>16,.2f}")
    print()

    print("=" * 78)
    print("TOP 10 CHAMPIONS (highest revenue)")
    print("=" * 78)
    champs = df[df["segment"] == "Champions"].nlargest(10, "monetary")
    for i, (_, row) in enumerate(champs.iterrows(), 1):
        print(f"  {i:2d}. Customer {int(row['customer_id']):>6d}  "
              f"GBP {row['monetary']:>12,.2f}  "
              f"{int(row['frequency'])} orders  "
              f"last {int(row['recency'])}d ago")
    print()

    print("=" * 78)
    print("TOP 10 AT RISK (high value, but haven't purchased recently)")
    print("=" * 78)
    at_risk = df[df["segment"] == "At Risk"].nlargest(10, "monetary")
    for i, (_, row) in enumerate(at_risk.iterrows(), 1):
        print(f"  {i:2d}. Customer {int(row['customer_id']):>6d}  "
              f"GBP {row['monetary']:>12,.2f}  "
              f"{int(row['frequency'])} orders  "
              f"last {int(row['recency'])}d ago")
    print()

    champ_pct = summary.loc[summary["segment"] == "Champions", "revenue_pct"].values[0]
    champ_cust_pct = summary.loc[summary["segment"] == "Champions", "customer_pct"].values[0]
    at_risk_rev = summary.loc[summary["segment"] == "At Risk", "revenue"].values[0]
    at_risk_cust = summary.loc[summary["segment"] == "At Risk", "customers"].values[0]

    print("=" * 78)
    print("KEY INSIGHTS")
    print("=" * 78)
    print(f"  Champions: {champ_cust_pct:.1f}% of customers, {champ_pct:.1f}% of revenue")
    print(f"  At Risk:   {at_risk_cust:,} customers, GBP {at_risk_rev:,.2f} revenue at stake")
    print()

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RFM Segmentation Summary", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Reference date: {ref_date.date()}",
        f"Total customers: {total_customers:,}",
        f"Total revenue: GBP {total_revenue:,.2f}", "",
        "## Segments", "",
        "| Segment | Customers | % Cust | Revenue (GBP) | % Rev | Avg Recency (d) | Avg Freq |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['segment']} | {row['customers']:,} | {row['customer_pct']}% | "
            f"{row['revenue']:,.2f} | {row['revenue_pct']}% | "
            f"{row['avg_recency_days']:.0f} | {row['avg_frequency']:.1f} |"
        )
    DOC_OUT.write_text("\n".join(lines) + "\n")

    con.close()
    print(f"Saved: {CSV_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
