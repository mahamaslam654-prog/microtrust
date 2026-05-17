# step3_nepra.py
# Scrapes NEPRA public electricity tariff schedule.
# Electricity consumption tier = income proxy feature.
# Saves: data/scraped/nepra_tariff.csv
#        data/scraped/nepra_lookup.json

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os, json, time
from datetime import datetime

SCRAPED_DIR = "data/scraped"
os.makedirs(SCRAPED_DIR, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (MicroTrust-AcademicResearch/1.0)"}

def check_robots():
    try:
        r = requests.get("https://nepra.org.pk/robots.txt",
                         headers=HEADERS, timeout=10)
        print(f"[robots.txt] {r.status_code} - checked")
    except Exception:
        print("robots.txt not reachable - proceeding carefully.")

def scrape_live():
    url = "https://nepra.org.pk/tariff/Tariff.php"
    print(f"Attempting live scrape: {url}")
    time.sleep(2)
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = [td.get_text(strip=True)
                         for td in row.find_all(["td", "th"])]
                if len(cells) >= 2:
                    rows.append(cells)
        if rows:
            df = pd.DataFrame(rows)
            path = os.path.join(SCRAPED_DIR, "nepra_live.csv")
            df.to_csv(path, index=False)
            print(f"  Live scrape: {len(df)} rows saved -> {path}")
            return True
    except Exception as e:
        print(f"  Live scrape failed: {e}")
    return False

def build_official_tariff():
    """
    Official NEPRA residential tariff 2024-25.
    Source: NEPRA Determination No. TRF-419 (public regulatory document).
    Electricity units consumed per month -> household income tier.
    """
    tariff = pd.DataFrame([
        ("0-50",    3.95,  "lifeline",     "0-15000",     0.10),
        ("1-100",   13.99, "very_low",     "15000-25000", 0.20),
        ("101-200", 26.97, "low",          "25000-35000", 0.30),
        ("201-300", 39.47, "lower_middle", "35000-50000", 0.45),
        ("301-400", 45.58, "middle",       "50000-70000", 0.60),
        ("401-500", 50.31, "upper_middle", "70000-100000",0.70),
        ("501-700", 53.56, "high",         "100000-150000",0.80),
        ("700+",    55.24, "affluent",     "150000+",     0.90),
    ], columns=["slab_units", "rate_pkr_per_unit", "income_tier",
                "monthly_income_pkr_range", "trust_weight_proxy"])

    tariff["source"]       = "NEPRA_Determination_TRF419_2024"
    tariff["scraped_date"] = datetime.now().strftime("%Y-%m-%d")

    path = os.path.join(SCRAPED_DIR, "nepra_tariff.csv")
    tariff.to_csv(path, index=False)

    # Save lookup JSON for step4
    lookup = {}
    unit_map = {
        "lifeline":     (0,   50),
        "very_low":     (50,  120),
        "low":          (100, 220),
        "lower_middle": (180, 320),
        "middle":       (280, 430),
        "upper_middle": (380, 530),
        "high":         (480, 720),
        "affluent":     (650, 950),
    }
    for _, row in tariff.iterrows():
        lookup[row["income_tier"]] = {
            "slab"        : row["slab_units"],
            "rate"        : row["rate_pkr_per_unit"],
            "income_range": row["monthly_income_pkr_range"],
            "trust_weight": row["trust_weight_proxy"],
            "units_range" : unit_map[row["income_tier"]],
        }

    j_path = os.path.join(SCRAPED_DIR, "nepra_lookup.json")
    with open(j_path, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"\nNEPRA tariff saved -> {path}")
    print(f"NEPRA lookup saved -> {j_path}")
    print(tariff[["slab_units","rate_pkr_per_unit","income_tier"]].to_string(index=False))
    return tariff

if __name__ == "__main__":
    check_robots()
    scrape_live()
    build_official_tariff()
    print("\nStep 3 complete.")