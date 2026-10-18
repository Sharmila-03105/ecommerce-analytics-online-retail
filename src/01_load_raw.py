from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
XLSX_PATH = RAW_DIR / "online_retail_II.xlsx"
PARQUET_PATH = RAW_DIR / "online_retail_II.parquet"
PROFILE_PATH = Path("docs/raw_profile.md")

SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sheets() -> pd.DataFrame:
    frames = []
    for sheet in SHEETS:
        print(f"  loading sheet: {sheet}")
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet, engine="openpyxl")
        df["sheet_source"] = sheet
        print(f"    rows: {len(df):,}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Force mixed-type columns to string (prevents pyarrow int/str clash)
    for col in ["Invoice", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "Customer ID" in df.columns:
        df["Customer ID"] = df["Customer ID"].astype("string")

    return df


def profile(df: pd.DataFrame) -> str:
    lines = [
        "# Raw Data Profile", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Source file: {XLSX_PATH.name}",
        f"SHA256: {sha256(XLSX_PATH)}", "",
        f"Total rows: {len(df):,}",
        f"Total columns: {df.shape[1]}", "",
        "## Columns", "",
        "| Column | Dtype | Non-null | Null % |", "|---|---|---|---|",
    ]
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_pct = (1 - non_null / len(df)) * 100
        lines.append(f"| {col} | {df[col].dtype} | {non_null:,} | {null_pct:.1f}% |")

    lines += ["", "## Row counts by sheet", "", "| Sheet | Rows |", "|---|---|"]
    for sheet, count in df["sheet_source"].value_counts().items():
        lines.append(f"| {sheet} | {count:,} |")

    if "InvoiceDate" in df.columns:
        dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        lines += ["", "## Date range", "", f"- Min: {dt.min()}", f"- Max: {dt.max()}"]

    return "\n".join(lines) + "\n"


def main() -> int:
    if not XLSX_PATH.exists():
        print(f"ERROR: {XLSX_PATH} not found.", file=sys.stderr)
        return 1

    print(f"Reading {XLSX_PATH} ...")
    df = load_sheets()

    print(f"Total rows loaded: {len(df):,}")
    print(f"Writing {PARQUET_PATH} ...")
    df.to_parquet(PARQUET_PATH, index=False, compression="snappy")

    print(f"Writing profile -> {PROFILE_PATH}")
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(profile(df))

    size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Parquet size: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
