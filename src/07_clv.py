"""Customer Lifetime Value using BG/NBD + Gamma-Gamma."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import warnings

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")

from lifetimes import BetaGeoFitter, GammaGammaFitter  # noqa: E402

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
CSV_OUT = Path("data/staging/customer_clv.csv")
DOC_OUT = Path("docs/clv_summary.md")


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Pulling customer transactions ...")

    # Per-customer RFM-style summary needed by BG/NBD:
    #   frequency: number of REPEAT purchases (not total invoices)
    #   recency:   days between first and last purchase
    #   T:         days between first purchase and reference date
    #   monetary_value: average revenue per invoice
    query = """
    WITH cust AS (
        SELECT
            CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS customer_id,
            invoice_no,
            invoice_date,
            revenue
        FROM fact_sales
        WHERE NOT is_cancellation
          AND NOT is_guest
          AND NOT is_non_product
          AND NOT is_invalid_price
          AND customer_id IS NOT NULL
    ),
    invoice_level AS (
        SELECT
            customer_id,
            invoice_no,
            MIN(invoice_date) AS inv_date,
            SUM(revenue) AS inv_revenue
        FROM cust
        GROUP BY customer_id, invoice_no
    ),
    summary AS (
        SELECT
            customer_id,
            MIN(inv_date) AS first_date,
            MAX(inv_date) AS last_date,
            COUNT(DISTINCT invoice_no) AS total_orders,
            AVG(inv_revenue) AS avg_order_value,
            SUM(inv_revenue) AS total_revenue
        FROM invoice_level
        GROUP BY customer_id
    )
    SELECT * FROM summary
    """
    df = con.execute(query).fetchdf()
    print(f"  customers: {len(df):,}")

    if df.empty:
        print("ERROR: no customers returned.")
        return 1

    ref_date = pd.to_datetime(df["last_date"]).max() + pd.Timedelta("1D")
    print(f"  reference date: {ref_date.date()}")

    first = pd.to_datetime(df["first_date"])
    last = pd.to_datetime(df["last_date"])

    # BG/NBD needs frequency = repeat purchases (total_orders - 1)
    df["frequency"] = (df["total_orders"] - 1).clip(lower=0)
    df["recency"] = (last - first).dt.days
    df["T"] = (ref_date - first).dt.days
    df["monetary_value"] = df["avg_order_value"]

    # BG/NBD requires frequency > 0 for customers to contribute to fit
    # We'll fit on repeat buyers, then score everyone
    print()
    print("Fitting BG/NBD model ...")
    repeat_df = df[df["frequency"] > 0].copy()
    print(f"  repeat buyers (frequency > 0): {len(repeat_df):,}")

    if len(repeat_df) < 100:
        print("ERROR: not enough repeat buyers to fit model.")
        con.close()
        return 1

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(repeat_df["frequency"], repeat_df["recency"], repeat_df["T"])
    print("  BG/NBD fit complete.")

    # Predict expected number of purchases in next 90 days
    print()
    print("Predicting future purchases ...")
    df["pred_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        df["frequency"],
        df["recency"],
        df["T"],
    )
    df["pred_purchases_12m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        365,
        df["frequency"],
        df["recency"],
        df["T"],
    )

    # Probability the customer is still "alive" (not churned)
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"], df["recency"], df["T"]
    )

    print(f"  median prob_alive: {df['prob_alive'].median():.2f}")

    # Gamma-Gamma only on customers with frequency > 0
    print()
    print("Fitting Gamma-Gamma model ...")
    gg_df = df[(df["frequency"] > 0) & (df["monetary_value"] > 0)].copy()
    print(f"  customers for Gamma-Gamma: {len(gg_df):,}")

    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(gg_df["frequency"], gg_df["monetary_value"])

    # Expected average order value for every customer
    df["expected_avg_order_value"] = ggf.conditional_expected_average_profit(
        df["frequency"].clip(lower=1),
        df["monetary_value"].fillna(gg_df["monetary_value"].mean()).clip(lower=0.01),
    )

    # CLV over next 12 months (discounted)
    print()
    print("Computing 12-month CLV ...")
    df["clv_12m"] = ggf.customer_lifetime_value(
        bgf,
        df["frequency"],
        df["recency"],
        df["T"],
        df["monetary_value"].fillna(gg_df["monetary_value"].mean()).clip(lower=0.01),
        time=12,
        discount_rate=0.01,
    )
    # Some CLV values come back negative or huge — clamp for sanity
    df["clv_12m"] = df["clv_12m"].clip(lower=0)

    # Segment the CLV scores
    df["clv_band"] = pd.qcut(
        df["clv_12m"].rank(method="first"),
        4,
        labels=["Low", "Medium", "High", "VIP"],
    ).astype(str)

    # Save back to warehouse
    out = df[[
        "customer_id", "total_orders", "avg_order_value", "total_revenue",
        "frequency", "recency", "T", "monetary_value",
        "prob_alive", "pred_purchases_90d", "pred_purchases_12m",
        "expected_avg_order_value", "clv_12m", "clv_band",
    ]].copy()
    out["customer_id"] = out["customer_id"].astype(int)

    con.execute("DROP TABLE IF EXISTS customer_clv")
    con.register("clv_df", out)
    con.execute("CREATE TABLE customer_clv AS SELECT * FROM clv_df")
    con.unregister("clv_df")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV_OUT, index=False)

    # Summary
    total_clv = out["clv_12m"].sum()
    print()
    print("=" * 78)
    print("CLV SUMMARY")
    print("=" * 78)
    print(f"  Total customers: {len(out):,}")
    print(f"  Total 12m CLV: GBP {total_clv:,.2f}")
    print(f"  Median CLV:    GBP {out['clv_12m'].median():,.2f}")
    print(f"  Mean CLV:      GBP {out['clv_12m'].mean():,.2f}")

    print()
    print("=" * 78)
    print("CLV BY BAND")
    print("=" * 78)
    band_summary = out.groupby("clv_band").agg(
        customers=("customer_id", "count"),
        total_clv=("clv_12m", "sum"),
        avg_clv=("clv_12m", "mean"),
    ).reindex(["VIP", "High", "Medium", "Low"])
    band_summary["clv_pct"] = (band_summary["total_clv"] / total_clv * 100).round(1)
    for band, row in band_summary.iterrows():
        print(
            f"  {band:<8} {int(row['customers']):>6,} customers   "
            f"GBP {row['total_clv']:>14,.2f}   "
            f"({row['clv_pct']:>5.1f}%)   "
            f"avg GBP {row['avg_clv']:>8,.2f}"
        )

    print()
    print("=" * 78)
    print("TOP 10 VIP CUSTOMERS BY CLV")
    print("=" * 78)
    top = out.nlargest(10, "clv_12m")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"  {i:2d}. Customer {int(row['customer_id']):>6d}   "
            f"12m CLV GBP {row['clv_12m']:>10,.2f}   "
            f"{int(row['total_orders'])} orders   "
            f"P(alive) {row['prob_alive']:.2f}"
        )

    print()
    print("=" * 78)
    print("HIGH-VALUE CUSTOMERS AT RISK (high CLV, low P(alive))")
    print("=" * 78)
    at_risk_vip = out[
        (out["clv_band"].isin(["VIP", "High"])) & (out["prob_alive"] < 0.5)
    ].nlargest(10, "clv_12m")
    for i, (_, row) in enumerate(at_risk_vip.iterrows(), 1):
        print(
            f"  {i:2d}. Customer {int(row['customer_id']):>6d}   "
            f"12m CLV GBP {row['clv_12m']:>10,.2f}   "
            f"P(alive) {row['prob_alive']:.2f}   "
            f"last purchase {int(row['recency'])}d after 1st"
        )

    revenue_at_risk = at_risk_vip["clv_12m"].sum()
    print()
    print(f"  Total 12m CLV at risk (top 10 shown): GBP {revenue_at_risk:,.2f}")

    # Docs
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CLV Summary", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Reference date: {ref_date.date()}", "",
        "## Model", "",
        "- BG/NBD: predicts future purchase count",
        "- Gamma-Gamma: predicts average order value",
        "- CLV = discounted expected purchases × expected value, 12-month horizon",
        "- Discount rate: 1% monthly", "",
        "## Totals", "",
        f"- Total customers: {len(out):,}",
        f"- Total 12-month CLV: GBP {total_clv:,.2f}",
        f"- Median CLV: GBP {out['clv_12m'].median():,.2f}",
        f"- Mean CLV: GBP {out['clv_12m'].mean():,.2f}", "",
        "## CLV bands", "",
        "| Band | Customers | Total CLV (GBP) | % CLV | Avg CLV (GBP) |",
        "|---|---|---|---|---|",
    ]
    for band, row in band_summary.iterrows():
        lines.append(
            f"| {band} | {int(row['customers']):,} | {row['total_clv']:,.2f} | "
            f"{row['clv_pct']}% | {row['avg_clv']:,.2f} |"
        )
    DOC_OUT.write_text("\n".join(lines) + "\n")

    con.close()
    print()
    print(f"Saved: {CSV_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
