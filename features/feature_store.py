# feature_store.py
# Builds a sklearn Pipeline that transforms raw data into model-ready features.
# 4 feature families: behavioral, transactional, digital, social/contextual
# Saves: models/pipeline.joblib

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

os.makedirs("models", exist_ok=True)

# ── Feature families ──────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Behavioral
    "payment_streak_months",
    "bills_paid_ontime_12mo",
    "bills_late_12mo",
    "utility_units_per_month",
    "monthly_bill_pkr",
    # Transactional
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "loan_tenor_months",
    # Digital
    "mobile_txn_per_month",
    # Demographic
    "person_age",
    "person_emp_length",
    "family_size",
    "dependents",
    "peer_defaults",
    "cb_cred_hist_length_years",
    # World Bank context
    "wb_account_ownership_pct",
    "wb_gdp_per_capita_usd",
    "wb_gini_index",
]

BINARY_FEATURES = [
    "has_mobile_wallet",
    "has_smartphone",
    "has_guarantor",
    "income_seasonal",
    "is_unbanked",
]

ORDINAL_FEATURES = [
    "nepra_income_tier",
]
ORDINAL_CATEGORIES = [
    ["lifeline", "very_low", "low", "lower_middle",
     "middle", "upper_middle", "high", "affluent"]
]

NOMINAL_FEATURES = [
    "occupation",
    "city",
    "urban_type",
    "house_ownership",
    "loan_intent",
    "wallet_provider",
    "gender",
    "cb_person_default_on_file",
]

# ── Engineered features transformer ──────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds 3 engineered features before the main pipeline runs:
    1. repayment_momentum  — weighted recent payment behavior
    2. financial_stress    — loan burden relative to income
    3. stability_score     — inverse of payment variability
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Repayment momentum (recent months weighted more)
        X["repayment_momentum"] = (
            X["payment_streak_months"] * 0.5 +
            X["bills_paid_ontime_12mo"] * 0.3 +
            (1 - X.get("bills_late_12mo", 0) / 12) * 0.2
        )

        # 2. Financial stress index
        X["financial_stress"] = X["loan_percent_income"].clip(0, 5)

        # 3. Stability score (higher = more stable)
        X["stability_score"] = (
            X["payment_streak_months"] / 12 * 0.6 +
            X["person_emp_length"] / 20 * 0.4
        ).clip(0, 1)

        return X


def get_feature_names():
    """Returns all feature names after engineering."""
    return (NUMERIC_FEATURES +
            ["repayment_momentum", "financial_stress", "stability_score"] +
            BINARY_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES)


def build_pipeline():
    """
    Builds and returns the full sklearn preprocessing pipeline.
    Steps:
      1. FeatureEngineer  — adds 3 engineered columns
      2. ColumnTransformer:
           numeric   -> StandardScaler
           binary    -> passthrough (already 0/1)
           ordinal   -> OrdinalEncoder (income tier has natural order)
           nominal   -> OneHotEncoder (occupation, city, etc.)
    """
    numeric_all = (NUMERIC_FEATURES +
                   ["repayment_momentum", "financial_stress", "stability_score"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_all),
            ("bin", "passthrough",   BINARY_FEATURES),
            ("ord", OrdinalEncoder(
                        categories=ORDINAL_CATEGORIES,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ), ORDINAL_FEATURES),
            ("nom", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ), NOMINAL_FEATURES),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("engineer",     FeatureEngineer()),
        ("preprocessor", preprocessor),
    ])

    return pipeline


def fit_and_save(pipeline, X_train):
    """Fits the pipeline on training data and saves it."""
    pipeline.fit(X_train)
    path = "models/pipeline.joblib"
    joblib.dump(pipeline, path)
    print(f"Pipeline saved -> {path}")
    return pipeline


if __name__ == "__main__":
    # Load training data
    train = pd.read_csv("data/processed/train.csv")
    X_train = train.drop(columns=["loan_status"])
    y_train = train["loan_status"]

    print(f"Training data: {X_train.shape}")
    print(f"Feature families:")
    print(f"  Numeric   : {len(NUMERIC_FEATURES)}")
    print(f"  Binary    : {len(BINARY_FEATURES)}")
    print(f"  Ordinal   : {len(ORDINAL_FEATURES)}")
    print(f"  Nominal   : {len(NOMINAL_FEATURES)}")
    print(f"  Engineered: 3 (repayment_momentum, financial_stress, stability_score)")

    pipeline = build_pipeline()
    pipeline = fit_and_save(pipeline, X_train)

    # Verify output shape
    X_transformed = pipeline.transform(X_train)
    print(f"\nInput shape  : {X_train.shape}")
    print(f"Output shape : {X_transformed.shape}")
    print("\nPhase 3 complete. Pipeline ready for model training.")