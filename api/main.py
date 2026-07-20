# api/main.py
# FastAPI backend — serves Trust Score predictions
# Run with: uvicorn api.main:app --reload
# POST /predict -> trust_score, risk_band, top 5 SHAP contributors

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle, joblib, sys, os

sys.path.insert(0, ".")
from features.feature_store import build_pipeline

app = FastAPI(title="MicroTrust API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and explainer on startup
model    = joblib.load("models/xgboost.joblib")
with open("models/shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# Refit pipeline on training data
train    = pd.read_csv("data/processed/train.csv")
X_train  = train.drop(columns=["loan_status"])
pipeline = build_pipeline()
pipeline.fit(X_train)


class Applicant(BaseModel):
    person_age               : float = 30
    person_income            : float = 360000
    person_emp_length        : float = 3.0
    family_size              : int   = 4
    dependents               : int   = 2
    loan_amnt                : float = 50000
    loan_int_rate            : float = 24.0
    loan_percent_income      : float = 0.14
    loan_tenor_months        : int   = 12
    payment_streak_months    : int   = 8
    bills_paid_ontime_12mo   : int   = 9
    bills_late_12mo          : int   = 3
    utility_units_per_month  : float = 150
    monthly_bill_pkr         : float = 4000
    mobile_txn_per_month     : int   = 10
    peer_defaults            : int   = 1
    cb_cred_hist_length_years: int   = 2
    has_mobile_wallet        : int   = 1
    has_smartphone           : int   = 1
    has_guarantor            : int   = 0
    income_seasonal          : int   = 0
    is_unbanked              : int   = 1
    wb_account_ownership_pct : float = 27.3
    wb_gdp_per_capita_usd    : float = 1478.0
    wb_gini_index            : float = 33.5
    wb_bank_branches_per_100k: float = 10.8
    occupation               : str   = "small shopkeeper"
    city                     : str   = "lahore"
    urban_type               : str   = "urban"
    house_ownership          : str   = "rent"
    loan_intent              : str   = "working_capital"
    wallet_provider          : str   = "jazzcash"
    gender                   : str   = "male"
    cb_person_default_on_file: str   = "n"
    nepra_income_tier        : str   = "low"
    data_source              : str   = "api_input"


@app.get("/")
def root():
    return {"message": "MicroTrust API is running"}


@app.post("/predict")
def predict(applicant: Applicant):
    # Convert to DataFrame
    data = pd.DataFrame([applicant.model_dump()])

    # Transform
    X = pipeline.transform(data)

    # Predict
    p_default   = float(model.predict_proba(X)[0][1])
    trust_score = round((1 - p_default) * 100, 1)

    # Risk band
    if trust_score >= 65:
        risk_band = "Low Risk"
        color     = "green"
    elif trust_score >= 40:
        risk_band = "Medium Risk"
        color     = "amber"
    else:
        risk_band = "High Risk"
        color     = "red"

    # SHAP top 5
    shap_vals = explainer(X).values[0]
    top5_idx  = np.argsort(np.abs(shap_vals))[::-1][:5]

    feature_names = [f"feature_{i}" for i in range(len(shap_vals))]
    top5 = [
        {
            "feature"   : feature_names[i],
            "shap_value": round(float(shap_vals[i]), 4),
            "direction" : "increases risk" if shap_vals[i] > 0 else "decreases risk"
        }
        for i in top5_idx
    ]

    return {
        "trust_score"  : trust_score,
        "risk_band"    : risk_band,
        "color"        : color,
        "p_default"    : round(p_default, 4),
        "top5_shap"    : top5,
        "recommendation": (
            "Recommend for micro-loan approval."
            if trust_score >= 65
            else "Further review recommended."
            if trust_score >= 40
            else "High default risk. Do not approve."
        )
    }