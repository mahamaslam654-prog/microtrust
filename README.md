# MicroTrust: Alternative Credit Risk Profiling

An AI-powered micro-finance risk assessment system for Pakistan's informal economy.

## SDG Alignment
- SDG 1 — No Poverty
- SDG 10 — Reduced Inequalities

## Live Demo

- **Backend API:** https://microtrust-1uvl.onrender.com

## Tech Stack
- Backend: FastAPI (Python)
- Frontend: React.js
- ML: XGBoost, Random Forest, Logistic Regression
- Explainability: SHAP

## Data Sources
- World Bank API (266 countries)
- SBP Quarterly Reports
- NEPRA Tariff Schedule
- Synthetic Pakistani profiles (calibrated)

## Results
| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression | 0.678 | 0.519 |
| Random Forest | 0.737 | 0.596 |
| XGBoost | 0.729 | 0.597 |

## Trust Score Formula
Trust Score = (1 - P_default) × 100

## Risk Bands
- 🟢 65-100: Low Risk
- 🟡 40-65: Medium Risk
- 🔴 0-40: High Risk

## How to Run

### Backend
```bash
source venv/bin/activate
uvicorn api.main:app --reload
```

### Frontend
```bash
cd app/microtrust-ui
npm start
```

## Project Structure
