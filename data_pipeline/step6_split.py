# step6_split.py
# Stratified 70/15/15 train-val-test split.
# Saves: data/processed/train.csv, val.csv, test.csv

import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROCESSED_DIR = "data/processed"

def split():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_clean.csv"))
    print(f"Input: {df.shape} | Default rate: {df['loan_status'].mean():.1%}\n")

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    print(f"{'Split':<8} {'Rows':>6}  {'Default%':>9}")
    print("-" * 28)
    for name, Xs, ys in [("train", X_train, y_train),
                          ("val",   X_val,   y_val),
                          ("test",  X_test,  y_test)]:
        out = Xs.copy()
        out["loan_status"] = ys.values
        out.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
        print(f"{name:<8} {len(out):>6}  {ys.mean():>8.1%}")

    print("\nFiles saved:")
    print("  train.csv <- train models on this")
    print("  val.csv   <- tune hyperparameters on this")
    print("  test.csv  <- touch ONLY for final evaluation")

if __name__ == "__main__":
    split()
    print("\nStep 6 complete.")