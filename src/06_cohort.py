"""Cohort retention: monthly acquisition cohorts x month-offset retention %."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
CSV_OUT = Path("data/staging/cohort_retention.csv")
MATRIX_OUT = Path("data/staging/cohort_matrix.csv")
DOC_OUT = Path("docs/cohort_retention.md")


def clean_id(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Building cohorts ...")

    # 1. Each customer's cohort month = month of first purchase
    # 2. Each customer's activity per month
    # 3. Join to get month_offset = months since first purchase
    query = """
    WITH cleaned AS (
        SELECT
            CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS customer_id,
            DATE_TRUNC('month', invoice_date)::DATE AS activity_month
        FROM fact_sales
        WHERE NOT is_cancellation
          AND NOT is_guest
          AND NOT is_non_product
          AND NOT is_invalid_price
          AND customer_id IS NOT NULL
    ),
    first_purchase AS (
        SELECT
            customer_id,
            MIN(activity_month) AS cohort_month
        FROM cleaned
        GROUP BY customer_id
    ),
    activity AS (
        SELECT DISTINCT
            c.customer_id,
            f.cohort_month,
            c.activity_month,
            DATEDIFF('month', f.cohort_month, c.activity_month) AS month_offset
        FROM cleaned c
        JOIN first_purchase f USING (customer_id)
        WHERE c.activity_month >= f.cohort_month
    ),
    cohort_size AS (
        SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_users
        FROM first_purchase
        GROUP BY cohort_month
    ),
    retention AS (
        SELECT
            a.cohort_month,
            a.month_offset,
            COUNT(DISTINCT a.customer_id) AS retained_users
        FROM activity a
        GROUP BY a.cohort_month, a.month_offset
    )
    SELECT
        r.cohort_month,
        r.month_offset,
        r.retained_users,
        cs.cohort_users,
        ROUND(100.0 * r.retained_users / cs.cohort_users, 2) AS retention_pct
    FROM retention r
    JOIN cohort_size cs USING (cohort_month)
    ORDER BY r.cohort_month, r.month_offset
    """

    df = con.execute(query).fetchdf()
    print(f"  retention rows: {len(df):,}")

    if df.empty:
        print("ERROR: no data produced.")
        return 1

    # Save raw retention data
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)

    # Build matrix: cohort_month x month_offset
    matrix = df.pivot_table(
        index="cohort_month",
        columns="month_offset",
        values="retention_pct",
        aggfunc="first",
    ).fillna(0)
    matrix = matrix.sort_index()
    matrix.to_csv(MATRIX_OUT)

    # Print terminal heatmap (first 12 offsets)
    print()
    print("=" * 110)
    print("COHORT RETENTION MATRIX (%)")
    print("=" * 110)
    header = "Cohort".ljust(14) + "Size".rjust(8)
    max_off = min(12, matrix.shape[1])
    for o in range(max_off):
        header += f"M{o}".rjust(7)
    print(header)
    print("-" * 110)

    cohort_sizes = (
        con.execute("""
            SELECT
                CAST(CAST(customer_id AS DOUBLE) AS BIGINT) AS cid,
                DATE_TRUNC('month', MIN(invoice_date))::DATE AS cohort
            FROM fact_sales
            WHERE NOT is_cancellation AND NOT is_guest AND NOT is_non_product
              AND NOT is_invalid_price AND customer_id IS NOT NULL
            GROUP BY 1
        """).fetchdf()
    )
    size_map = cohort_sizes.groupby("cohort").size().to_dict()

    for cohort, row in matrix.iterrows():
        size = size_map.get(pd.Timestamp(cohort).date(), 0)
        line = str(cohort)[:10].ljust(14) + f"{size:>8,}"
        for o in range(max_off):
            val = row.get(o, 0) if o in row.index else 0
            line += f"{val:>6.1f}%"
        print(line)

    print()
    print("=" * 110)
    print("KEY STATISTICS")
    print("=" * 110)

    # Average retention by month offset (M1, M2, M3, M6, M12)
    for offset in [1, 2, 3, 6, 12]:
        val = df[df["month_offset"] == offset]["retention_pct"].mean()
        if pd.notna(val):
            print(f"  Month {offset:<3} avg retention: {val:.1f}%")

    # Best and worst cohorts by M1 retention
    m1 = df[df["month_offset"] == 1].merge(
        cohort_sizes.groupby("cohort").size().reset_index(name="size"),
        left_on="cohort_month",
        right_on="cohort",
    ).drop(columns=["cohort"])
    if not m1.empty:
        best = m1.loc[m1["retention_pct"].idxmax()]
        worst = m1.loc[m1["retention_pct"].idxmin()]
        print()
        print(f"  Best  M1 cohort: {best['cohort_month']}  ({best['retention_pct']:.1f}%)")
        print(f"  Worst M1 cohort: {worst['cohort_month']}  ({worst['retention_pct']:.1f}%)")

    print()

    # Docs
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cohort Retention Analysis", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Cohorts: {matrix.shape[0]}",
        f"Max month offset: {matrix.shape[1] - 1}", "",
        "## Retention matrix (%)", "",
    ]
    header_md = "| Cohort | Size |"
    for o in range(max_off):
        header_md += f" M{o} |"
    lines.append(header_md)
    sep = "|---|---|" + "---|" * max_off
    lines.append(sep)
    for cohort, row in matrix.iterrows():
        size = size_map.get(pd.Timestamp(cohort).date(), 0)
        line = f"| {cohort} | {size} |"
        for o in range(max_off):
            val = row.get(o, 0) if o in row.index else 0
            line += f" {val:.1f}% |"
        lines.append(line)

    DOC_OUT.write_text("\n".join(lines) + "\n")

    con.close()
    print(f"Saved: {CSV_OUT}")
    print(f"Saved: {MATRIX_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
