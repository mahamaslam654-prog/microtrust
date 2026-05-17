# ml/baseline.py
# Model 1: Logistic Regression baseline
# Saves: models/logistic_regression.joblib
#        models/metrics_lr.json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, brier_score_loss,
                              confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib, json, os, sys

sys.path.insert(0, ".")
from features.feature_store import build_pipeline

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

def load_data():
    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")
    X_train = train.drop(columns=["loan_status"])
    y_train = train["loan_status"]
    X_val   = val.drop(columns=["loan_status"])
    y_val   = val["loan_status"]
    return X_train, y_train, X_val, y_val

def evaluate(name, y_true, y_pred, y_prob):
    ks = max(
        y_prob[y_true == 0].mean() - y_prob[y_true == 1].mean(),
        y_prob[y_true == 1].mean() - y_prob[y_true == 0].mean()
    )
    metrics = {
        "model"    : name,
        "auc_roc"  : round(roc_auc_score(y_true, y_prob), 4),
        "f1"       : round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall"   : round(recall_score(y_true, y_pred), 4),
        "brier"    : round(brier_score_loss(y_true, y_prob), 4),
        "ks_stat"  : round(ks, 4),
    }
    print(f"\n{'='*40}")
    print(f"  {name} — Validation Metrics")
    print(f"{'='*40}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(f"\n{classification_report(y_true, y_pred)}")
    return metrics

def plot_roc(y_true, y_prob, name):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#e74c3c", lw=2)
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/roc_{name.lower().replace(' ','_')}.png")
    plt.close()
    print(f"ROC curve saved -> plots/roc_{name.lower().replace(' ','_')}.png")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()

    # Build and fit pipeline
    pipeline = build_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t   = pipeline.transform(X_val)

    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        C=0.1
    )
    lr.fit(X_train_t, y_train)

    # Predict
    y_pred = lr.predict(X_val_t)
    y_prob = lr.predict_proba(X_val_t)[:, 1]

    # Evaluate
    metrics = evaluate("Logistic Regression", y_val, y_pred, y_prob)

    # Save
    joblib.dump(lr, "models/logistic_regression.joblib")
    joblib.dump(pipeline, "models/pipeline_lr.joblib")
    with open("models/metrics_lr.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_val, y_prob, "Logistic Regression")

    print("\nModel saved -> models/logistic_regression.joblib")
    print("Metrics saved -> models/metrics_lr.json")
    print("\nBaseline complete.")