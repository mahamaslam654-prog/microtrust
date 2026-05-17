# eda.py — Full EDA, saves all plots to plots/ folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (10, 5)
sns.set_theme(style="whitegrid")

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/processed/merged_clean.csv")
print(f"Shape: {df.shape}")
print(f"Default rate: {df['loan_status'].mean():.1%}\n")

# 1. Class balance
fig, ax = plt.subplots()
counts = df["loan_status"].value_counts()
bars = ax.bar(["Repaid (0)", "Defaulted (1)"], counts.values,
              color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{val} ({val/len(df):.1%})", ha="center", fontsize=11)
ax.set_title("Target Class Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("plots/01_class_balance.png")
plt.close()
print("Saved: 01_class_balance.png")

# 2. Default rate by occupation
occ_default = df.groupby("occupation")["loan_status"].mean().sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(occ_default.index, occ_default.values * 100, color="#e67e22", edgecolor="white")
ax.axvline(df["loan_status"].mean() * 100, color="red", linestyle="--", label="Overall avg")
ax.set_xlabel("Default Rate (%)")
ax.set_title("Default Rate by Occupation")
ax.legend()
plt.tight_layout()
plt.savefig("plots/02_default_by_occupation.png")
plt.close()
print("Saved: 02_default_by_occupation.png")

# 3. Default rate by city
city_default = df.groupby("city")["loan_status"].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
ax.bar(city_default.index, city_default.values * 100, color="#3498db", edgecolor="white")
ax.axhline(df["loan_status"].mean() * 100, color="red", linestyle="--", label="Overall avg")
ax.set_ylabel("Default Rate (%)")
ax.set_title("Default Rate by City")
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_default_by_city.png")
plt.close()
print("Saved: 03_default_by_city.png")

# 4. Payment streak vs default
fig, ax = plt.subplots()
df.boxplot(column="payment_streak_months", by="loan_status", ax=ax)
ax.set_title("Payment Streak by Loan Status")
ax.set_xlabel("Loan Status (0=Repaid, 1=Defaulted)")
ax.set_ylabel("Payment Streak (months)")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plots/04_payment_streak_vs_default.png")
plt.close()
print("Saved: 04_payment_streak_vs_default.png")

# 5. Correlation heatmap
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
            ax=ax, linewidths=0.3)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/05_correlation_heatmap.png")
plt.close()
print("Saved: 05_correlation_heatmap.png")

# 6. Feature distributions
key_features = ["person_income","loan_amnt","loan_int_rate",
                "payment_streak_months","bills_paid_ontime_12mo",
                "utility_units_per_month","mobile_txn_per_month"]
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    if col in df.columns:
        df[col].hist(ax=axes[i], bins=30, color="#9b59b6", edgecolor="white")
        axes[i].set_title(col, fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distribution of Key Features")
plt.tight_layout()
plt.savefig("plots/06_feature_distributions.png")
plt.close()
print("Saved: 06_feature_distributions.png")

# 7. Mobile wallet vs default
wallet_default = df.groupby("has_mobile_wallet")["loan_status"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["No Wallet", "Has Wallet"], wallet_default.values * 100,
       color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.4)
ax.set_ylabel("Default Rate (%)")
ax.set_title("Default Rate: Mobile Wallet vs No Wallet")
plt.tight_layout()
plt.savefig("plots/07_wallet_vs_default.png")
plt.close()
print("Saved: 07_wallet_vs_default.png")

# Top correlations with target
print("\nTop 10 features correlated with loan_status:")
print(corr["loan_status"].drop("loan_status")
          .abs().sort_values(ascending=False).head(10))

print("\nEDA complete. All plots saved to plots/")