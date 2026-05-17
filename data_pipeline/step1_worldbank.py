# step1_worldbank.py
# Pulls financial inclusion indicators from World Bank API (free, no login)
# Saves: data/scraped/worldbank_all_countries.csv
#        data/scraped/pakistan_wb_profile.json
#        data/scraped/regional_benchmark.csv

import wbdata
import pandas as pd
import os
import json
import time

SCRAPED_DIR = "data/scraped"
os.makedirs(SCRAPED_DIR, exist_ok=True)

INDICATORS = {
    "FX.OWN.TOTL.ZS"    : "account_ownership_pct",
    "FX.OWN.TOTL.FE.ZS" : "account_ownership_female_pct",
    "FB.CBK.BRCH.P5"    : "bank_branches_per_100k",
    "NY.GDP.PCAP.CD"    : "gdp_per_capita_usd",
    "SI.POV.GINI"       : "gini_index",
    "SL.TLF.CACT.ZS"   : "labor_force_participation_pct",
}

PEER_COUNTRIES = ["PAK","IND","BGD","NPL","PHL","KEN","NGA","EGY","IDN","LKA"]

def fetch_indicators():
    print("Fetching World Bank indicators (this takes ~60 seconds)...")
    all_frames = []

    for wb_code, col_name in INDICATORS.items():
        try:
            print(f"  Fetching: {col_name}...")
            data = wbdata.get_dataframe({wb_code: col_name}, country="all")
            data = data.reset_index()
            data = data.sort_values("date", ascending=False)
            data = data.groupby("country").first().reset_index()
            all_frames.append(data[["country", col_name]])
            time.sleep(1)
        except Exception as e:
            print(f"  Warning: could not fetch {col_name}: {e}")
            continue

    if not all_frames:
        print("No data fetched. Check internet connection.")
        return None

    from functools import reduce
    merged = reduce(lambda l, r: pd.merge(l, r, on="country", how="outer"), all_frames)

    out = os.path.join(SCRAPED_DIR, "worldbank_all_countries.csv")
    merged.to_csv(out, index=False)
    print(f"\nSaved {len(merged)} countries -> {out}")
    return merged

def extract_pakistan(df):
    pak = df[df["country"] == "Pakistan"].copy()

    if pak.empty:
        print("Pakistan not found, using hardcoded SBP values.")
        profile = {
            "country": "Pakistan",
            "account_ownership_pct": 21.3,
            "account_ownership_female_pct": 7.9,
            "bank_branches_per_100k": 10.8,
            "gini_index": 29.6,
            "source": "SBP_FSR_2024_fallback"
        }
    else:
        profile = pak.iloc[0].dropna().to_dict()
        profile["source"] = "WorldBank_API"

    out = os.path.join(SCRAPED_DIR, "pakistan_wb_profile.json")
    with open(out, "w") as f:
        json.dump(profile, f, indent=2)

    print("\nPakistan profile:")
    for k, v in profile.items():
        print(f"  {k}: {v}")
    return profile

def regional_benchmark(df):
    peers = df[df["country"].isin([
        "Pakistan","India","Bangladesh","Nepal",
        "Philippines","Kenya","Nigeria","Egypt, Arab Rep.",
        "Indonesia","Sri Lanka"
    ])].copy()

    out = os.path.join(SCRAPED_DIR, "regional_benchmark.csv")
    peers.to_csv(out, index=False)
    print(f"\nRegional benchmark saved -> {out}")
    print(peers[["country","account_ownership_pct","gdp_per_capita_usd"]].to_string(index=False))

if __name__ == "__main__":
    df = fetch_indicators()
    if df is not None:
        extract_pakistan(df)
        regional_benchmark(df)
    print("\nStep 1 complete.")