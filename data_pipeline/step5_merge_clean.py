# step5_merge_clean.py
# Merges all scraped sources, cleans, and saves final dataset.
# Saves: data/processed/merged_clean.csv

import pandas as pd
import numpy as np
import os, json

SCRAPED_DIR   = "data/scraped"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load():
    path = os.path.join(SCRAPED_DIR, "synthetic_pk_profiles.csv")
    df   = pd.read_csv(path)
    print(f"Loaded synthetic profiles: {df.shape}")

    # Add World Bank Pakistan context as scalar columns
    wb_path = os.path.join(SCRAPED_DIR, "pakistan_wb_profile.json")
    if os.path.exists(wb_path):
        with open(wb_path) as f:
            wb = json.load(f)
        df["wb_account_ownership_pct"]    = wb.get("account_ownership_pct", 27.3)
        df["wb_bank_branches_per_100k"]   = wb.get("bank_branches_per_100k", 10.8)
        df["wb_gdp_per_capita_usd"]       = wb.get("gdp_per_capita_usd", 1478.0)
        df["wb_gini_index"]               = wb.get("gini_index", 33.5)
        print(f"World Bank features added: 4 columns")

    return df

def clean(df):
    print(f"\nCleaning {len(df)} rows...")

    # 1. Drop missing targets
    before = len(df)
    df = df.dropna(subset=["loan_status"])
    print(f"  Dropped {before - len(df)} rows with missing target")

    # 2. Lowercase text columns
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).str.lower().str.strip()

    # 3. Fill missing numerics with median
    for c in df.select_dtypes(include=np.number).columns:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())

    # 4. Fill missing categoricals with mode
    for c in df.select_dtypes("object").columns:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].mode()[0])

    # 5. IQR outlier clipping
    for c in ["person_age","person_income","loan_amnt",
              "loan_int_rate","loan_percent_income","person_emp_length"]:
        if c not in df.columns:
            continue
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR    = Q3 - Q1
        before = len(df)
        df     = df[(df[c] >= Q1 - 1.5*IQR) & (df[c] <= Q3 + 1.5*IQR)]
        if len(df) < before:
            print(f"  {c}: removed {before - len(df)} outliers")

    # 6. Fix types
    df["loan_status"]      = df["loan_status"].astype(int)
    df["has_mobile_wallet"]= df["has_mobile_wallet"].astype(int)

    print(f"\nClean shape   : {df.shape}")
    print(f"Default rate  : {df['loan_status'].mean():.1%}")
    print(f"Columns       : {len(df.columns)}")
    return df

if __name__ == "__main__":
    df  = load()
    df  = clean(df)
    out = os.path.join(PROCESSED_DIR, "merged_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("\nStep 5 complete.")