"""Export warehouse tables to CSVs for Power BI import."""
from pathlib import Path

import duckdb

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
OUT_DIR = Path("data/staging/powerbi")

TABLES = [
    "fact_sales", "dim_customer", "dim_product", "dim_date", "dim_country",
    "customer_rfm", "customer_clv", "churn_scores",
    "revenue_monthly", "revenue_forecast", "customer_survival",
    "market_basket_rules",
]


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    print(f"Exporting to {OUT_DIR} ...")
    for table in TABLES:
        try:
            out_path = OUT_DIR / f"{table}.csv"
            con.execute(
                f"COPY (SELECT * FROM {table}) TO '{out_path.as_posix()}' "
                f"(FORMAT CSV, HEADER TRUE)"
            )
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  {table:<24} {n:>10,} rows   {size_mb:>6.2f} MB")
        except Exception as e:
            print(f"  {table:<24} FAILED: {e}")

    con.close()
    print(f"\nDone. CSVs in: {OUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
