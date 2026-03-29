# clean_and_merge.py
# loads dominick's UPC lookup + movement data, merges, computes per-unit price
# outputs scratch/detergent_clean.csv

import pandas as pd
import numpy as np
import re
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "scratch")
os.makedirs(SCRATCH, exist_ok=True)

# load UPC lookup
upc = pd.read_csv(os.path.join(ROOT, "upclnd.csv"), encoding="latin-1")
print(f"UPC lookup: {len(upc)} products")


def parse_size(s):
    """Convert '290 OZ', '25 LB', '150 EA' to (numeric_oz, unit_flag)."""
    if pd.isna(s):
        return np.nan, None
    s = str(s).strip().upper()
    m = re.match(r"([\d.]+)\s*(OZ|LB|EA|CT|GAL)", s)
    if not m:
        return np.nan, None
    val, unit = float(m.group(1)), m.group(2)
    if unit == "LB":
        return val * 16.0, "OZ"
    elif unit == "GAL":
        return val * 128.0, "OZ"
    elif unit == "OZ":
        return val, "OZ"
    else:
        return val, unit


parsed = upc["SIZE"].apply(parse_size)
upc["size_oz"] = parsed.apply(lambda x: x[0])
upc["size_unit"] = parsed.apply(lambda x: x[1])

# keep only OZ-convertible products
upc_oz = upc[upc["size_unit"] == "OZ"].copy()
print(f"UPCs with OZ-convertible size: {len(upc_oz)} / {len(upc)}")

# brand extraction from DESCRIP field
BRAND_MAP = {
    "TIDE": "Tide", "ALL ": "All", "ALL-": "All", "ULTRA ALL": "All",
    "WISK": "Wisk", "SURF": "Surf", "ERA ": "Era", "CHEER": "Cheer",
    "GAIN": "Gain", "PUREX": "Purex", "ARM": "Arm & Hammer",
    "DYNAMO": "Dynamo", "FAB ": "Fab", "AJAX": "Ajax", "YES ": "Yes",
    "BOLD": "Bold", "DASH": "Dash", "IVORY": "Ivory", "DREFT": "Dreft",
    "OXYDOL": "Oxydol", "SOLO": "Solo", "XTRA": "Xtra", "RINSO": "Rinso",
    "OMEGA": "Omega", "CLOROX": "Clorox", "SNUGGLE": "Snuggle",
    "LEVER": "Lever", "FRESH START": "Fresh Start",
}


def extract_brand(descrip):
    d = re.sub(r"^[~$*]+", "", str(descrip)).upper()
    for key, brand in BRAND_MAP.items():
        if key in d:
            return brand
    return "Other"


upc_oz["brand"] = upc_oz["DESCRIP"].apply(extract_brand)
print(f"Brand distribution in UPC table:\n{upc_oz['brand'].value_counts().head(10)}\n")

# load movement data
print("Loading wlnd.csv (~6.7M rows)...")
mov = pd.read_csv(
    os.path.join(ROOT, "wlnd.csv"),
    usecols=["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "OK"],
    dtype={"STORE": "int32", "UPC": "int64", "WEEK": "int16",
           "MOVE": "int32", "QTY": "int16", "OK": "int8"},
    low_memory=False,
    encoding="latin-1",
)
print(f"Loaded: {len(mov):,} rows")

# drop invalid obs
mov = mov[(mov["PRICE"] > 0) & (mov["MOVE"] > 0) & (mov["OK"] == 1)].copy()
print(f"After filters (PRICE>0, MOVE>0, OK=1): {len(mov):,} rows")

# merge movement with UPC lookup
df = mov.merge(upc_oz[["UPC", "DESCRIP", "size_oz", "brand"]], on="UPC", how="inner")
print(f"After merge: {len(df):,} rows")

# per-unit price (QTY handles multi-unit deals like 2-for-$5.99)
df["unit_price"] = df["PRICE"] / df["QTY"]
df["ppu"] = df["unit_price"] / df["size_oz"]
df["ppu_cents"] = df["ppu"] * 100
df["log_size"] = np.log(df["size_oz"])

# sale flag
df["sale_flag"] = df["SALE"].notna().astype(int)

# drop remaining invalid and trim extreme ppu (likely data errors)
df = df.dropna(subset=["ppu_cents", "size_oz"])
q_low, q_high = df["ppu_cents"].quantile(0.001), df["ppu_cents"].quantile(0.999)
n_before = len(df)
df = df[(df["ppu_cents"] >= q_low) & (df["ppu_cents"] <= q_high)]
print(f"Trimmed extremes (0.1% tails): dropped {n_before - len(df):,} rows")

# save
out_cols = ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "unit_price",
            "DESCRIP", "size_oz", "ppu", "ppu_cents", "log_size",
            "brand", "sale_flag"]
df[out_cols].to_csv(os.path.join(SCRATCH, "detergent_clean.csv"), index=False)

print(f"\nFinal dataset: {len(df):,} obs | {df['UPC'].nunique()} UPCs | "
      f"{df['STORE'].nunique()} stores | {df['WEEK'].nunique()} weeks")
print(f"\nBrand counts (top 10):\n{df['brand'].value_counts().head(10)}")
print(f"\nSize (oz) summary:\n{df['size_oz'].describe()}")
print(f"\nPPU (cents/oz) summary:\n{df['ppu_cents'].describe()}")
print(f"\nSaved to scratch/detergent_clean.csv")
