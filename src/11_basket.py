"""Market basket analysis using FP-Growth association rules."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

DB_PATH = Path("data/warehouse/ecommerce.duckdb")
CSV_OUT = Path("data/staging/market_basket_rules.csv")
DOC_OUT = Path("docs/market_basket.md")

MAX_INVOICES = 20000
MIN_SUPPORT = 0.01
MIN_LIFT = 1.2


def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found.")
        return 1

    con = duckdb.connect(str(DB_PATH))

    print("Pulling invoice-product pairs ...")
    query = f"""
    SELECT f.invoice_no, p.description
    FROM fact_sales f
    JOIN dim_product p ON f.stock_code = p.stock_code
    WHERE NOT f.is_cancellation
      AND NOT f.is_guest
      AND NOT f.is_non_product
      AND NOT f.is_invalid_price
      AND p.description IS NOT NULL
      AND f.invoice_no IN (
          SELECT invoice_no FROM fact_sales
          WHERE NOT is_cancellation AND NOT is_guest AND NOT is_non_product
          GROUP BY invoice_no
          HAVING COUNT(DISTINCT stock_code) BETWEEN 2 AND 30
          LIMIT {MAX_INVOICES}
      )
    """
    df = con.execute(query).fetchdf()
    print(f"  pairs: {len(df):,}")
    print(f"  invoices: {df['invoice_no'].nunique():,}")
    print(f"  products: {df['description'].nunique():,}")

    if df.empty:
        print("ERROR: no data.")
        return 1

    print("Grouping into baskets ...")
    baskets = df.groupby("invoice_no")["description"].apply(list).tolist()
    print(f"  baskets: {len(baskets):,}")

    print("Encoding ...")
    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_array, columns=te.columns_)

    print("Running FP-Growth ...")
    frequent = fpgrowth(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
    print(f"  frequent itemsets: {len(frequent):,}")

    if frequent.empty:
        print("ERROR: no frequent itemsets.")
        con.close()
        return 1

    print("Generating rules ...")
    rules = association_rules(frequent, metric="lift", min_threshold=MIN_LIFT)
    rules = rules.sort_values("lift", ascending=False)
    print(f"  rules: {len(rules):,}")

    rules["antecedent"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequent"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    display = rules[["antecedent", "consequent", "support", "confidence", "lift"]].copy()

    print()
    print("=" * 110)
    print("TOP 15 ASSOCIATION RULES BY LIFT")
    print("=" * 110)
    print(f"{'Antecedent':<45}{'Consequent':<40}{'Support':>8}{'Conf':>7}{'Lift':>7}")
    print("-" * 110)
    for _, row in display.head(15).iterrows():
        print(f"{row['antecedent'][:43]:<45}{row['consequent'][:38]:<40}"
              f"{row['support']:>8.3f}{row['confidence']:>7.3f}{row['lift']:>7.2f}")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(CSV_OUT, index=False)

    con.execute("DROP TABLE IF EXISTS market_basket_rules")
    con.register("rules_df", display)
    con.execute("CREATE TABLE market_basket_rules AS SELECT * FROM rules_df")
    con.unregister("rules_df")

    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Market Basket Analysis", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Baskets: {len(baskets):,}",
        f"Frequent itemsets: {len(frequent):,}",
        f"Rules: {len(rules):,}", "",
        "## Top 15 rules by lift", "",
        "| Antecedent | Consequent | Support | Confidence | Lift |",
        "|---|---|---|---|---|",
    ]
    for _, row in display.head(15).iterrows():
        lines.append(f"| {row['antecedent']} | {row['consequent']} | "
                     f"{row['support']:.3f} | {row['confidence']:.3f} | {row['lift']:.2f} |")
    lines += ["", "## Business use", "",
              "- Cross-sell recommendations",
              "- Bundle pricing",
              "- 'Frequently bought together' widgets"]
    DOC_OUT.write_text("\n".join(lines) + "\n")

    con.close()
    print()
    print(f"Saved: {CSV_OUT}")
    print(f"Saved: {DOC_OUT}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
