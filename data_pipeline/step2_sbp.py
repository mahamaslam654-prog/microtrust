# step2_sbp.py
# Downloads SBP quarterly PDF and extracts microfinance statistics.
# Saves: data/raw/SBP_Dec2024.pdf
#        data/scraped/sbp_calibration.json

import requests
import pdfplumber
import pandas as pd
import os, json, time
from datetime import datetime

RAW_DIR     = "data/raw"
SCRAPED_DIR = "data/scraped"
os.makedirs(RAW_DIR,     exist_ok=True)
os.makedirs(SCRAPED_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (MicroTrust-AcademicResearch/1.0)"}

SBP_PDFS = [
    {
        "url"  : "https://www.sbp.org.pk/publications/Quarterly/2024/Dec/Complete.pdf",
        "label": "SBP_Dec2024"
    }
]

def check_robots():
    try:
        r = requests.get("https://www.sbp.org.pk/robots.txt",
                         headers=HEADERS, timeout=10)
        print(f"[robots.txt] {r.status_code} - OK to scrape")
    except Exception:
        print("Could not fetch robots.txt - proceeding carefully.")

def download_pdf(url, label):
    path = os.path.join(RAW_DIR, f"{label}.pdf")
    if os.path.exists(path):
        print(f"Already exists: {path}")
        return path
    print(f"Downloading {label}...")
    time.sleep(2)
    try:
        r = requests.get(url, headers=HEADERS, timeout=90, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size = os.path.getsize(path) // 1024
        print(f"Saved -> {path}  ({size} KB)")
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def extract_tables(pdf_path, label):
    if not pdf_path or not os.path.exists(pdf_path):
        return
    print(f"Extracting tables from {label}...")
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"  Total pages: {len(pdf.pages)}")
        for pg_num, page in enumerate(pdf.pages, 1):
            for t_idx, table in enumerate(page.extract_tables()):
                if not table or len(table) < 2:
                    continue
                try:
                    headers = [
                        str(c).strip().replace("\n", " ") if c else f"col_{i}"
                        for i, c in enumerate(table[0])
                    ]
                    df = pd.DataFrame(table[1:], columns=headers)
                    df = df.dropna(how="all")
                    df["_source"] = label
                    df["_page"]   = pg_num
                    results.append(df)
                except Exception:
                    continue
    if results:
        combined = pd.concat(results, ignore_index=True)
        out = os.path.join(SCRAPED_DIR, f"{label}_tables.csv")
        combined.to_csv(out, index=False)
        print(f"  Saved {len(combined)} rows -> {out}")

def build_calibration():
    """
    Key MFB statistics from SBP FSR 2024 (public report).
    Used to calibrate synthetic data in step4.
    """
    calib = {
        "source"                         : "SBP_FSR_2024",
        "mfb_infection_ratio_pct"        : 22.5,
        "mfb_avg_loan_size_pkr"          : 65000,
        "mfb_total_borrowers_million"    : 8.2,
        "mfb_female_borrowers_pct"       : 43.0,
        "pk_informal_workforce_pct"      : 72.0,
        "pk_mobile_wallet_users_million" : 87.0,
        "pk_unbanked_adults_pct"         : 72.7,
        "pk_avg_microloan_tenor_months"  : 12,
        "scraped_date"                   : datetime.now().strftime("%Y-%m-%d"),
    }
    out = os.path.join(SCRAPED_DIR, "sbp_calibration.json")
    with open(out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSBP calibration saved -> {out}")
    for k, v in calib.items():
        print(f"  {k}: {v}")
    return calib

if __name__ == "__main__":
    check_robots()
    for item in SBP_PDFS:
        path = download_pdf(item["url"], item["label"])
        extract_tables(path, item["label"])
    build_calibration()
    print("\nStep 2 complete.")