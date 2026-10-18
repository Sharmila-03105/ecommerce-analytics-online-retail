import duckdb
con = duckdb.connect("data/warehouse/ecommerce.duckdb")
n = con.execute("SELECT COUNT(*) FROM churn_scores").fetchone()[0]
bands = con.execute("SELECT churn_risk, COUNT(*) FROM churn_scores GROUP BY churn_risk").fetchall()
print(f"churn_scores: {n:,} rows")
for band, count in bands:
    print(f"  {band}: {count:,}")
con.close()
