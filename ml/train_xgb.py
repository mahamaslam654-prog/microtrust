# ml/train_xgb.py
# Model 3: XGBoost tuned with Optuna
# Saves: models/xgboost.joblib
#        models/metrics_xgb.json

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, brier_score_loss,
                              confusion_matrix, classification_report)
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import joblib, json, os, sys

optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    ks = abs(float(y_prob[y_true == 0].mean()) - float(y_prob[y_true == 1].mean()))
    metrics = {
        "model"    : name,
        "auc_roc"  : round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1"       : round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall"   : round(float(recall_score(y_true, y_pred)), 4),
        "brier"    : round(float(brier_score_loss(y_true, y_prob)), 4),
        "ks_stat"  : round(float(ks), 4),
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
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#8e44ad", lw=2)
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

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    def objective(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 400),
            "max_depth"       : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma"           : trial.suggest_float("gamma", 0, 5),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "random_state"    : 42,
            "eval_metric"     : "auc",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_t, y_train,
                  eval_set=[(X_val_t, y_val)],
                  verbose=False)
        y_prob = model.predict_proba(X_val_t)[:, 1]
        return float(roc_auc_score(y_val, y_prob))

    print("Tuning XGBoost with Optuna (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nBest trial AUC : {study.best_value:.4f}")
    print(f"Best params    : {study.best_params}")

    best_params = {k: v for k, v in study.best_params.items()}
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"]     = 42
    best_params["eval_metric"]      = "auc"

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_t, y_train,
                    eval_set=[(X_val_t, y_val)],
                    verbose=False)

    y_pred = final_model.predict(X_val_t)
    y_prob = final_model.predict_proba(X_val_t)[:, 1]

    metrics = evaluate("XGBoost", y_val, y_pred, y_prob)

    joblib.dump(final_model, "models/xgboost.joblib")
    joblib.dump(pipeline,    "models/pipeline_xgb.joblib")

    with open("models/metrics_xgb.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_val, y_prob, "XGBoost")

    print("\nModel saved    -> models/xgboost.joblib")
    print("Metrics saved  -> models/metrics_xgb.json")
    print("\nXGBoost complete.")