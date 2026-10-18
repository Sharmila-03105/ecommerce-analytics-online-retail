import duckdb
con = duckdb.connect("data/warehouse/ecommerce.duckdb")

print("=" * 60)
print("CHURN RISK BANDS")
print("=" * 60)
rows = con.execute("""
    SELECT
        churn_risk,
        COUNT(*) AS customers,
        ROUND(AVG(churn_probability), 3) AS avg_prob,
        ROUND(SUM(monetary), 2) AS revenue_at_stake
    FROM churn_scores
    GROUP BY churn_risk
    ORDER BY avg_prob DESC
""").fetchall()
for risk, cust, prob, rev in rows:
    print(f"  {risk:<8} {cust:>5,} customers   avg P(churn) {prob:.3f}   GBP {rev:>12,.2f}")

print()
print("=" * 60)
print("TOP 10 HIGH-VALUE CUSTOMERS AT HIGH CHURN RISK")
print("=" * 60)
rows = con.execute("""
    SELECT customer_id, churn_probability, monetary, frequency, recency_days
    FROM churn_scores
    WHERE churn_risk = 'High'
    ORDER BY monetary DESC
    LIMIT 10
""").fetchall()
for i, (cid, prob, mon, freq, rec) in enumerate(rows, 1):
    print(f"  {i:2d}. Customer {int(cid):>6d}   P(churn) {prob:.2f}   "
          f"GBP {mon:>10,.2f}   {int(freq)} orders   {int(rec)}d ago")

con.close()
