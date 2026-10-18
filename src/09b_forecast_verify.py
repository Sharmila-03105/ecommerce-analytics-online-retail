import duckdb
con = duckdb.connect("data/warehouse/ecommerce.duckdb")

print("=" * 70)
print("MONTHLY REVENUE HISTORY (last 12 months)")
print("=" * 70)
rows = con.execute("""
    SELECT month, ROUND(revenue, 2) AS revenue
    FROM revenue_monthly
    ORDER BY month DESC
    LIMIT 12
""").fetchall()
for m, r in rows:
    print(f"  {m}   GBP {r:>14,.2f}")

print()
print("=" * 70)
print("FORECAST (next 6 months)")
print("=" * 70)
rows = con.execute("""
    SELECT month, ROUND(forecast_revenue, 2) AS f, model
    FROM revenue_forecast
    ORDER BY month
""").fetchall()
for m, f, model in rows:
    print(f"  {m}   GBP {f:>14,.2f}   ({model})")

con.close()
