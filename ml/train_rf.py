# ml/train_rf.py
# Model 2: Random Forest
# Saves: models/random_forest.joblib
#        models/metrics_rf.json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, brier_score_loss,
                              confusion_matrix, classification_report)
from sklearn.metrics import roc_curve
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
    ks = abs(y_prob[y_true == 0].mean() - y_prob[y_true == 1].mean())
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
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#27ae60", lw=2)
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend()
    plt.tight_layout()
    fname = f"plots/roc_{name.lower().replace(' ','_')}.png"
    plt.savefig(fname)
    plt.close()
    print(f"ROC curve saved -> {fname}")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_data()

    pipeline = build_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t   = pipeline.transform(X_val)

    print("Tuning Random Forest with GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth"   : [4, 6, 8],
        "min_samples_leaf": [5, 10],
    }
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )
    grid = GridSearchCV(rf, param_grid, cv=5,
                        scoring="roc_auc", n_jobs=-1, verbose=1)
    grid.fit(X_train_t, y_train)

    best_rf = grid.best_estimator_
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV AUC: {grid.best_score_:.4f}")

    y_pred = best_rf.predict(X_val_t)
    y_prob = best_rf.predict_proba(X_val_t)[:, 1]

    metrics = evaluate("Random Forest", y_val, y_pred, y_prob)

    joblib.dump(best_rf, "models/random_forest.joblib")
    joblib.dump(pipeline, "models/pipeline_rf.joblib")
    with open("models/metrics_rf.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_val, y_prob, "Random Forest")

    print("\nModel saved -> models/random_forest.joblib")
    print("Metrics saved -> models/metrics_rf.json")
    print("\nRandom Forest complete.")