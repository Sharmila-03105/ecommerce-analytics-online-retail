import duckdb
con = duckdb.connect("data/warehouse/ecommerce.duckdb")
try:
    n = con.execute("SELECT COUNT(*) FROM customer_clv").fetchone()[0]
    print(f"customer_clv rows: {n:,}")
    sample = con.execute("SELECT * FROM customer_clv LIMIT 3").fetchdf()
    print(sample[["customer_id", "clv_12m", "prob_alive", "clv_band"]])
except Exception as e:
    print(f"CLV table missing or error: {e}")
con.close()
