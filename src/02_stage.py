import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/online_retail_II.parquet")
STG = Path("data/staging/stg_transactions.parquet")
LOG = Path("docs/cleaning_log.md")

NON_PROD = {"POST","D","M","C2","DOT","BANK CHARGES","PADS","AMAZONFEE","CRUK","S","B","GIFT","ADJUST","ADJUST2"}
COUNTRY_MAP = {"USA":"United States","RSA":"South Africa","EIRE":"Ireland"}
COL_MAP = {"Invoice":"invoice_no","InvoiceNo":"invoice_no","StockCode":"stock_code",
           "Description":"description","Quantity":"quantity","InvoiceDate":"invoice_date",
           "Price":"unit_price","UnitPrice":"unit_price","Customer ID":"customer_id",
           "CustomerID":"customer_id","Country":"country"}

def main():
    if not RAW.exists():
        print(f"ERROR: {RAW} not found. Run 01_load_raw.py first.", file=sys.stderr)
        return 1
    print(f"Reading {RAW} ...")
    df = pd.read_parquet(RAW)
    before = len(df)
    print(f"  rows: {before:,}")

    df = df.rename(columns=COL_MAP)
    df["is_cancellation"] = df["invoice_no"].astype(str).str.startswith("C")
    df["is_adjustment"] = (df["quantity"] < 0) & (~df["is_cancellation"])
    df["is_guest"] = df["customer_id"].isna()

    def np_code(c):
        if pd.isna(c): return True
        s = str(c).strip().upper()
        return s in NON_PROD or s.startswith(("DCGS","TEST","GIFT"))
    df["is_non_product"] = df["stock_code"].apply(np_code)

    df["is_invalid_price"] = df["unit_price"] <= 0
    df["is_missing_desc"] = df["description"].isna()
    df["country"] = df["country"].replace(COUNTRY_MAP)
    df["description"] = df["description"].astype("string").str.strip().str.replace(r"\s+"," ",regex=True)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    bd = len(df)
    df = df.drop_duplicates()
    dropped = bd - len(df)
    after = len(df)

    log = ["# Cleaning Log", "",
           f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
           f"Input: {before:,}", f"Output: {after:,}", f"Dropped: {dropped:,}", "",
           "| Rule | Flag | Count |", "|---|---|---|",
           f"| Cancellations | is_cancellation | {df['is_cancellation'].sum():,} |",
           f"| Adjustments | is_adjustment | {df['is_adjustment'].sum():,} |",
           f"| Guests | is_guest | {df['is_guest'].sum():,} |",
           f"| Non-product | is_non_product | {df['is_non_product'].sum():,} |",
           f"| Invalid price | is_invalid_price | {df['is_invalid_price'].sum():,} |",
           f"| Missing desc | is_missing_desc | {df['is_missing_desc'].sum():,} |",
           f"| Duplicates | (dropped) | {dropped:,} |"]

    cols = ["invoice_no","stock_code","description","quantity","invoice_date","unit_price",
            "customer_id","country","sheet_source","is_cancellation","is_adjustment",
            "is_guest","is_non_product","is_invalid_price","is_missing_desc"]
    df = df[cols]

    STG.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(STG, index=False, compression="snappy")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(log))
    print(f"Done. Staging rows: {after:,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
