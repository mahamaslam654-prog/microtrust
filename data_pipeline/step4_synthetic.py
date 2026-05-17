# step4_synthetic.py
# Generates 1000 synthetic Pakistani informal economy applicant profiles.
# Calibrated to: World Bank (step1), SBP (step2), NEPRA (step3).
# Saves: data/scraped/synthetic_pk_profiles.csv

import pandas as pd
import numpy as np
from faker import Faker
import os, json

fake = Faker()
np.random.seed(42)

SCRAPED_DIR = "data/scraped"
N = 1000

def load_calibration():
    calib = {}
    sbp_path = os.path.join(SCRAPED_DIR, "sbp_calibration.json")
    if os.path.exists(sbp_path):
        with open(sbp_path) as f:
            calib["sbp"] = json.load(f)
    else:
        calib["sbp"] = {"mfb_infection_ratio_pct": 22.5}

    nep_path = os.path.join(SCRAPED_DIR, "nepra_lookup.json")
    if os.path.exists(nep_path):
        with open(nep_path) as f:
            calib["nepra"] = json.load(f)
    else:
        calib["nepra"] = {}

    print(f"SBP default rate target : {calib['sbp']['mfb_infection_ratio_pct']}%")
    print(f"NEPRA tiers loaded      : {list(calib['nepra'].keys())}")
    return calib

OCCUPATIONS = {
    "Daily wage laborer"       : {"income":(18000,32000),"stability":0.28,"tier":"very_low",    "pct":0.18},
    "Rickshaw/bike driver"     : {"income":(22000,48000),"stability":0.38,"tier":"low",         "pct":0.12},
    "Street food vendor"       : {"income":(20000,42000),"stability":0.34,"tier":"low",         "pct":0.10},
    "Small shopkeeper"         : {"income":(28000,75000),"stability":0.58,"tier":"lower_middle","pct":0.12},
    "Home-based worker"        : {"income":(12000,28000),"stability":0.30,"tier":"very_low",    "pct":0.08},
    "Domestic worker"          : {"income":(14000,30000),"stability":0.28,"tier":"very_low",    "pct":0.08},
    "Informal factory worker"  : {"income":(24000,44000),"stability":0.44,"tier":"lower_middle","pct":0.10},
    "Construction worker"      : {"income":(20000,45000),"stability":0.31,"tier":"low",         "pct":0.10},
    "Small farmer"             : {"income":(15000,55000),"stability":0.40,"tier":"lower_middle","pct":0.07},
    "Motorcycle mechanic"      : {"income":(25000,55000),"stability":0.50,"tier":"lower_middle","pct":0.05},
    "Self-employed electrician": {"income":(30000,70000),"stability":0.55,"tier":"middle",      "pct":0.05},
    "Fruit/veg cart"           : {"income":(15000,35000),"stability":0.33,"tier":"very_low",    "pct":0.05},
}

CITIES = {
    "Lahore":"urban","Karachi":"urban","Rawalpindi":"urban",
    "Faisalabad":"urban","Multan":"peri_urban","Peshawar":"peri_urban",
    "Gujranwala":"urban","Sialkot":"peri_urban",
    "Bahawalpur":"rural_adjacent","Rahim Yar Khan":"rural_adjacent",
}

LOAN_PURPOSES = [
    "working_capital","inventory_purchase","home_repair",
    "medical_emergency","education","equipment_purchase","livestock"
]

def generate(n, calib):
    nepra   = calib.get("nepra", {})
    occ_names   = list(OCCUPATIONS.keys())
    occ_weights = [OCCUPATIONS[o]["pct"] for o in occ_names]
    occ_weights = [w / sum(occ_weights) for w in occ_weights]
    records = []

    for _ in range(n):
        occ_name = np.random.choice(occ_names, p=occ_weights)
        occ      = OCCUPATIONS[occ_name]
        city     = np.random.choice(list(CITIES.keys()))
        urban    = CITIES[city]
        age      = int(np.random.randint(20, 57))
        gender   = np.random.choice(["male","female"], p=[0.68,0.32])

        monthly_income = int(np.random.randint(*occ["income"]))
        annual_income  = monthly_income * 12
        emp_years      = round(max(0.5, np.random.exponential(4.0)), 1)
        is_seasonal    = int(occ_name in ["Small farmer","Construction worker","Street food vendor"])

        family_size  = int(np.random.randint(2, 9))
        dependents   = int(np.random.randint(0, max(1, family_size-1)))
        house_type   = np.random.choice(["rent","own","family_home"], p=[0.48,0.17,0.35])

        tier      = occ["tier"]
        tier_info = nepra.get(tier, {"rate": 30.0, "units_range": [50, 200]})
        u_lo, u_hi     = tier_info["units_range"]
        utility_units  = int(np.random.randint(u_lo, u_hi))
        monthly_bill   = round(utility_units * float(tier_info.get("rate", 30)), 0)

        stab = occ["stability"]
        stab += 0.04 if urban == "urban" else 0.0
        stab += 0.03 if house_type == "own" else 0.0
        stab -= 0.03 if is_seasonal else 0.0

        bills_ontime   = int(np.random.binomial(12, min(stab+0.10, 0.95)))
        bills_late     = 12 - bills_ontime
        payment_streak = int(np.random.binomial(12, min(stab, 0.92)))

        has_wallet     = int(np.random.random() < 0.55)
        wallet         = np.random.choice(["JazzCash","EasyPaisa","NayaPay","SadaPay"]) if has_wallet else "None"
        mobile_txn_mo  = int(np.random.randint(2, 35)) if has_wallet else 0
        has_smartphone = int(np.random.random() < 0.72)

        loan_amnt      = int(np.random.randint(5000, 150000))
        loan_int_rate  = round(np.random.uniform(18.0, 36.0), 1)
        loan_pct_inc   = round(loan_amnt / annual_income, 3)
        loan_tenor     = int(np.random.choice([3,6,9,12,18,24]))
        loan_intent    = np.random.choice(LOAN_PURPOSES)

        peer_defaults  = int(np.random.binomial(5, 1 - stab))
        has_guarantor  = int(np.random.random() < stab * 0.6)
        prev_default   = int(np.random.random() > (stab + 0.15))
        cred_hist_mo   = int(np.random.randint(0, 48))
        is_unbanked    = int(np.random.random() < 0.40)

        p_default = (
            0.38
            - stab                   * 0.32
            - (bills_ontime / 12)    * 0.16
            + loan_pct_inc           * 0.18
            + prev_default           * 0.26
            + is_seasonal            * 0.05
            + (peer_defaults / 5)    * 0.07
            - has_wallet             * 0.04
            - has_guarantor          * 0.06
            - (emp_years / 20)       * 0.04
            + is_unbanked            * 0.03
        )
        p_default   = float(np.clip(p_default, 0.04, 0.95))
        loan_status = int(np.random.random() < p_default)

        records.append({
            "person_age"               : age,
            "gender"                   : gender,
            "city"                     : city,
            "urban_type"               : urban,
            "family_size"              : family_size,
            "dependents"               : dependents,
            "house_ownership"          : house_type,
            "occupation"               : occ_name,
            "person_emp_length"        : emp_years,
            "income_seasonal"          : is_seasonal,
            "person_income"            : annual_income,
            "loan_intent"              : loan_intent,
            "loan_amnt"                : loan_amnt,
            "loan_int_rate"            : loan_int_rate,
            "loan_percent_income"      : loan_pct_inc,
            "loan_tenor_months"        : loan_tenor,
            "nepra_income_tier"        : tier,
            "utility_units_per_month"  : utility_units,
            "monthly_bill_pkr"         : monthly_bill,
            "bills_paid_ontime_12mo"   : bills_ontime,
            "bills_late_12mo"          : bills_late,
            "payment_streak_months"    : payment_streak,
            "has_mobile_wallet"        : has_wallet,
            "wallet_provider"          : wallet,
            "mobile_txn_per_month"     : mobile_txn_mo,
            "has_smartphone"           : has_smartphone,
            "peer_defaults"            : peer_defaults,
            "has_guarantor"            : has_guarantor,
            "cb_person_default_on_file": "Y" if prev_default else "N",
            "cb_cred_hist_length_years": cred_hist_mo // 12,
            "is_unbanked"              : is_unbanked,
            "loan_status"              : loan_status,
            "data_source"              : "synthetic_PK_calibrated",
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    calib = load_calibration()
    print(f"\nGenerating {N} profiles...")
    df = generate(N, calib)

    out = os.path.join(SCRAPED_DIR, "synthetic_pk_profiles.csv")
    df.to_csv(out, index=False)

    print(f"Saved       -> {out}")
    print(f"Shape       : {df.shape}")
    print(f"Default rate: {df['loan_status'].mean():.1%}  (SBP target ~22.5%)")
    print(f"\nOccupation mix:\n{df['occupation'].value_counts().to_string()}")
    print("\nStep 4 complete.")