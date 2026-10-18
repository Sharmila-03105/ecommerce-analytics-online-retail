from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

STG_PATH = Path("data/staging/stg_transactions.parquet")
DB_PATH = Path("data/warehouse/ecommerce.duckdb")


def build_warehouse(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> dict:
    stats = {}

    # ---------- fact_sales ----------
    fact = df.copy()
    fact["revenue"] = fact["quantity"] * fact["unit_price"]
    fact = fact.reset_index(drop=True)
    fact["sale_key"] = fact.index + 1

    fact_cols = [
        "sale_key", "invoice_no", "stock_code", "quantity",
        "invoice_date", "unit_price", "revenue", "customer_id", "country",
        "is_cancellation", "is_adjustment", "is_guest",
        "is_non_product", "is_invalid_price", "is_missing_desc",
    ]
    fact = fact[fact_cols]
    stats["fact_sales_rows"] = len(fact)

    # ---------- dim_customer ----------
    cust = (
        df[~df["is_guest"]]
        .groupby("customer_id", as_index=False)
        .agg(
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            total_orders=("invoice_no", "nunique"),
            total_revenue=("quantity", "size"),  # placeholder, recompute below
            country=("country", "first"),
        )
    )
    # Real revenue per customer
    rev = (
        df[~df["is_guest"]]
        .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
        .groupby("customer_id", as_index=False)["revenue"].sum()
    )
    cust = cust.drop(columns=["total_revenue"]).merge(rev, on="customer_id", how="left")
    cust = cust.reset_index(drop=True)
    cust["customer_key"] = cust.index + 1
    cust = cust[[
        "customer_key", "customer_id", "country",
        "first_purchase", "last_purchase", "total_orders", "revenue",
    ]]
    stats["dim_customer_rows"] = len(cust)

    # ---------- dim_product ----------
    prod = (
        df[~df["is_non_product"]]
        .groupby("stock_code", as_index=False)
        .agg(
            description=("description", "first"),
            times_sold=("invoice_no", "nunique"),
        )
    )
    rev_prod = (
        df[~df["is_non_product"]]
        .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
        .groupby("stock_code", as_index=False)["revenue"].sum()
    )
    prod = prod.merge(rev_prod, on="stock_code", how="left")
    prod = prod.reset_index(drop=True)
    prod["product_key"] = prod.index + 1
    prod = prod[["product_key", "stock_code", "description", "times_sold", "revenue"]]
    stats["dim_product_rows"] = len(prod)

    # ---------- dim_country ----------
    ctry = df[["country"]].drop_duplicates().reset_index(drop=True)
    ctry["country_key"] = ctry.index + 1
    ctry = ctry[["country_key", "country"]]
    stats["dim_country_rows"] = len(ctry)

    # ---------- dim_date ----------
    dates = pd.DataFrame({"date": pd.to_datetime(df["invoice_date"].dropna().unique())})
    dates["date_key"] = dates["date"].dt.strftime("%Y%m%d").astype(int)
    dates["year"] = dates["date"].dt.year
    dates["quarter"] = dates["date"].dt.quarter
    dates["month"] = dates["date"].dt.month
    dates["month_name"] = dates["date"].dt.month_name()
    dates["week"] = dates["date"].dt.isocalendar().week.astype(int)
    dates["weekday"] = dates["date"].dt.day_name()
    dates["is_weekend"] = dates["date"].dt.weekday >= 5
    dates = dates[[
        "date_key", "date", "year", "quarter", "month",
        "month_name", "week", "weekday", "is_weekend",
    ]]
    stats["dim_date_rows"] = len(dates)

    # ---------- Load into DuckDB ----------
    con.execute("DROP TABLE IF EXISTS fact_sales")
    con.execute("DROP TABLE IF EXISTS dim_customer")
    con.execute("DROP TABLE IF EXISTS dim_product")
    con.execute("DROP TABLE IF EXISTS dim_country")
    con.execute("DROP TABLE IF EXISTS dim_date")

    con.register("fact_df", fact)
    con.register("cust_df", cust)
    con.register("prod_df", prod)
    con.register("ctry_df", ctry)
    con.register("date_df", dates)

    con.execute("CREATE TABLE dim_country AS SELECT * FROM ctry_df")
    con.execute("CREATE TABLE dim_customer AS SELECT * FROM cust_df")
    con.execute("CREATE TABLE dim_product AS SELECT * FROM prod_df")
    con.execute("CREATE TABLE dim_date AS SELECT * FROM date_df")
    con.execute("CREATE TABLE fact_sales AS SELECT * FROM fact_df")

    for t in ["fact_df", "cust_df", "prod_df", "ctry_df", "date_df"]:
        con.unregister(t)

    return stats


def main() -> int:
    if not STG_PATH.exists():
        print(f"ERROR: {STG_PATH} not found. Run 02_stage.py first.", file=sys.stderr)
        return 1

    print(f"Reading {STG_PATH} ...")
    df = pd.read_parquet(STG_PATH)
    print(f"  rows: {len(df):,}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    print(f"Building warehouse -> {DB_PATH} ...")
    con = duckdb.connect(str(DB_PATH))

    stats = build_warehouse(df, con)

    print("\nWarehouse built:")
    for name, count in stats.items():
        print(f"  {name}: {count:,}")

    # Quick sanity queries
    print("\nSanity check:")
    total_rev = con.execute(
        "SELECT ROUND(SUM(revenue), 2) FROM fact_sales "
        "WHERE NOT is_cancellation AND NOT is_invalid_price AND NOT is_non_product"
    ).fetchone()[0]
    print(f"  Total net revenue: GBP {total_rev:,.2f}")

    top_country = con.execute(
        "SELECT country, ROUND(SUM(revenue), 2) AS rev FROM fact_sales "
        "WHERE NOT is_cancellation AND NOT is_invalid_price AND NOT is_non_product "
        "GROUP BY country ORDER BY rev DESC LIMIT 1"
    ).fetchone()
    print(f"  Top country: {top_country[0]} (GBP {top_country[1]:,.2f})")

    date_range = con.execute(
        "SELECT MIN(date), MAX(date) FROM dim_date"
    ).fetchone()
    print(f"  Date range: {date_range[0]} -> {date_range[1]}")

    con.close()
    print(f"\nDone. Database size: {DB_PATH.stat().st_size / (1024*1024):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
