# ml/explain.py
# SHAP explanations for XGBoost model
# Saves: models/shap_explainer.pkl
#        plots/shap_beeswarm.png
#        plots/shap_waterfall.png

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib, pickle, os, sys

sys.path.insert(0, ".")
from features.feature_store import build_pipeline

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test  = pd.read_csv("data/processed/test.csv")
    X_train = train.drop(columns=["loan_status"])
    X_test  = test.drop(columns=["loan_status"])
    y_test  = test["loan_status"]
    return X_train, X_test, y_test

if __name__ == "__main__":
    X_train, X_test, y_test = load_data()

    # Load model and refit pipeline on train
    model    = joblib.load("models/xgboost.joblib")
    pipeline = build_pipeline()
    pipeline.fit(X_train)

    X_train_t = pipeline.transform(X_train)
    X_test_t  = pipeline.transform(X_test)

    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_test_t)

    # Save explainer
    with open("models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print("Explainer saved -> models/shap_explainer.pkl")

    # Get feature names from pipeline
    try:
        ohe        = pipeline.named_steps["preprocessor"].named_transformers_["nom"]
        ohe_names  = ohe.get_feature_names_out(
            ["occupation","city","urban_type","house_ownership",
             "loan_intent","wallet_provider","gender",
             "cb_person_default_on_file"]
        ).tolist()
    except Exception:
        ohe_names = []

    numeric_all = [
        "payment_streak_months","bills_paid_ontime_12mo","bills_late_12mo",
        "utility_units_per_month","monthly_bill_pkr","person_income",
        "loan_amnt","loan_int_rate","loan_percent_income","loan_tenor_months",
        "mobile_txn_per_month","person_age","person_emp_length","family_size",
        "dependents","peer_defaults","cb_cred_hist_length_years",
        "wb_account_ownership_pct","wb_gdp_per_capita_usd","wb_gini_index",
        "repayment_momentum","financial_stress","stability_score"
    ]
    binary_features  = ["has_mobile_wallet","has_smartphone","has_guarantor",
                        "income_seasonal","is_unbanked"]
    ordinal_features = ["nepra_income_tier"]
    feature_names    = numeric_all + binary_features + ordinal_features + ohe_names

    # Trim to match actual output size
    n_features    = X_test_t.shape[1]
    feature_names = feature_names[:n_features]
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names)}")

    # 1. Beeswarm plot (global importance)
    shap_vals_arr = shap_values.values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals_arr, X_test_t,
        feature_names=feature_names,
        show=False, max_display=15
    )
    plt.title("SHAP Feature Importance (Beeswarm)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", bbox_inches="tight")
    plt.close()
    print("Saved -> plots/shap_beeswarm.png")

    # 2. Waterfall plot (single applicant explanation)
    idx = 0  # first test applicant
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title("SHAP Waterfall — Single Applicant", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/shap_waterfall.png", bbox_inches="tight")
    plt.close()
    print("Saved -> plots/shap_waterfall.png")

    # 3. Top 10 features by mean absolute SHAP
    mean_shap = np.abs(shap_vals_arr).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:10]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_shap[top_idx]

    plt.figure(figsize=(9, 5))
    plt.barh(top_names[::-1], top_vals[::-1], color="#8e44ad", edgecolor="white")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 10 Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig("plots/shap_top10.png")
    plt.close()
    print("Saved -> plots/shap_top10.png")

    print("\nTop 10 features:")
    for name, val in zip(top_names, top_vals):
        print(f"  {name:<35} {val:.4f}")

    # Save trust score for first 5 test applicants
    probs        = model.predict_proba(X_test_t)[:, 1]
    trust_scores = (1 - probs) * 100
    print("\nSample Trust Scores (first 5 test applicants):")
    for i, (ts, actual) in enumerate(zip(trust_scores[:5], y_test[:5])):
        band = "Low Risk" if ts >= 65 else "Medium Risk" if ts >= 40 else "High Risk"
        print(f"  Applicant {i+1}: Trust Score={ts:.1f} | Band={band} | Actual={'Default' if actual==1 else 'Repaid'}")

    print("\nSHAP complete.")