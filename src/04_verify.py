import duckdb

con = duckdb.connect("data/warehouse/ecommerce.duckdb")

print("=" * 60)
print("TABLES IN WAREHOUSE")
print("=" * 60)
tables = con.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_name
""").fetchall()
for (t,) in tables:
    count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:20s}  {count:>10,} rows")

print()
print("=" * 60)
print("TOP 10 PRODUCTS BY REVENUE (excluding cancellations, non-products, invalid prices)")
print("=" * 60)
rows = con.execute("""
    SELECT
        p.description,
        ROUND(SUM(f.revenue), 2) AS revenue
    FROM fact_sales f
    JOIN dim_product p USING (stock_code)
    WHERE NOT is_cancellation
      AND NOT is_invalid_price
      AND NOT is_non_product
    GROUP BY p.description
    ORDER BY revenue DESC
    LIMIT 10
""").fetchall()
for i, (desc, rev) in enumerate(rows, 1):
    print(f"  {i:2d}. {desc[:50]:50s}  GBP {rev:>12,.2f}")

print()
print("=" * 60)
print("REVENUE BY COUNTRY (top 10)")
print("=" * 60)
rows = con.execute("""
    SELECT
        country,
        ROUND(SUM(revenue), 2) AS revenue,
        COUNT(DISTINCT invoice_no) AS orders
    FROM fact_sales
    WHERE NOT is_cancellation
      AND NOT is_invalid_price
      AND NOT is_non_product
    GROUP BY country
    ORDER BY revenue DESC
    LIMIT 10
""").fetchall()
for i, (country, rev, orders) in enumerate(rows, 1):
    print(f"  {i:2d}. {country:30s}  GBP {rev:>14,.2f}  ({orders:,} orders)")

print()
print("=" * 60)
print("MONTHLY REVENUE TREND (last 12 months)")
print("=" * 60)
rows = con.execute("""
    SELECT
        d.year,
        d.month,
        ROUND(SUM(f.revenue), 2) AS revenue,
        COUNT(DISTINCT f.invoice_no) AS orders
    FROM fact_sales f
    JOIN dim_date d ON f.invoice_date::DATE = d.date
    WHERE NOT is_cancellation
      AND NOT is_invalid_price
      AND NOT is_non_product
    GROUP BY d.year, d.month
    ORDER BY d.year DESC, d.month DESC
    LIMIT 12
""").fetchall()
for year, month, rev, orders in rows:
    print(f"  {year}-{month:02d}  GBP {rev:>14,.2f}  ({orders:,} orders)")

print()
print("=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)
q = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(CASE WHEN is_cancellation THEN 1 ELSE 0 END) AS cancellations,
        SUM(CASE WHEN is_guest THEN 1 ELSE 0 END) AS guests,
        SUM(CASE WHEN is_non_product THEN 1 ELSE 0 END) AS non_products,
        SUM(CASE WHEN is_invalid_price THEN 1 ELSE 0 END) AS invalid_prices
    FROM fact_sales
""").fetchone()
print(f"  Total rows:       {q[0]:>10,}")
print(f"  Cancellations:    {q[1]:>10,}")
print(f"  Guest customers:  {q[2]:>10,}")
print(f"  Non-products:     {q[3]:>10,}")
print(f"  Invalid prices:   {q[4]:>10,}")

con.close()
print()
print("Done.")
