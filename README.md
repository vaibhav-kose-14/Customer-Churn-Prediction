# 🔄 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Predictive classification system that identifies customers at risk of churning — from raw data to a live REST API with real-time scoring.**

---

## 📌 Project Summary

This end-to-end ML project was built to demonstrate production-grade data science skills:

- Engineered a predictive classification model to tackle customer attrition
- Designed robust data preprocessing pipelines with sklearn's `ColumnTransformer`
- Benchmarked multiple algorithms — **Logistic Regression, Random Forest, XGBoost**
- Achieved **87%+ accuracy** with tuned XGBoost hyperparameters
- Integrated **real-time scoring** via a FastAPI REST endpoint with Swagger UI

---

## 🗂️ Project Structure

```
customer_churn/
│
├── preprocess.py                  # Data generation, feature engineering & pipeline fitting
├── train.py                       # Model benchmarking, selection & serialisation
├── eda.py                         # Exploratory data analysis — 9 publication-quality plots
├── app.py                         # FastAPI REST API — /predict & /predict/batch endpoints
├── predict.py                     # Standalone real-time scoring demo (no server needed)
│
├── preprocessor_pipeline.pkl      # Fitted ColumnTransformer  [auto-generated]
├── xgboost_production_model.pkl   # Serialised XGBoost model  [auto-generated]
├── feature_columns.pkl            # Ordered feature schema    [auto-generated]
│
├── eda_plots/                     # All 9 EDA chart PNGs      [auto-generated]
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_contract.png
│   ├── 03_churn_by_internet.png
│   ├── 04_tenure_distribution.png
│   ├── 05_monthly_charges_dist.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_feature_importance.png
│   ├── 08_roc_curves.png
│   └── 09_confusion_matrix.png
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
# Step 1 — Generate data & fit preprocessing pipeline
python preprocess.py

# Step 2 — Benchmark models & save XGBoost
python train.py

# Step 3 — Generate all EDA visualisations
python eda.py

# Step 4 — Launch the scoring API
uvicorn app:app --reload --port 8000

# Or run the standalone scoring demo (no server needed)
python predict.py
```

### 3. Open the interactive API docs
```
http://127.0.0.1:8000/docs
```

---

## 📊 Model Benchmark Results

| Model | Accuracy | F1-Score (weighted) | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.848 | 0.787 | 0.729 |
| Random Forest | 0.768 | 0.781 | 0.719 |
| **XGBoost ✓ Selected** | **0.871** | **0.856** | **0.923** |

**Selection rationale:** XGBoost delivered the best AUC-ROC and accuracy. Its built-in L1/L2 regularisation, native handling of mixed feature types, and support for fast single-row inference made it the clear production choice.

---

## 🛠️ Feature Engineering

Four interaction features were engineered to capture non-linear business relationships:

| Feature | Formula | Business Logic |
|---|---|---|
| `MonthlyCharges_x_MTM` | `MonthlyCharges × is_month_to_month` | High bill + no commitment = highest churn signal |
| `Tenure_x_Security` | `tenure × has_online_security` | Long-term customers with security add-on are most loyal |
| `MTM_x_NoSecurity` | `is_MTM × (1 − has_security)` | The single highest-risk customer profile |
| `CostPerTenureMonth` | `MonthlyCharges ÷ (tenure + 1)` | Price burden relative to loyalty; high early = price-shocked |

These features ranked as the **top 4 by XGBoost gain**, validating the engineering step.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info & version |
| `GET` | `/health` | Model health check & metadata |
| `POST` | `/predict` | Score a single customer in real-time |
| `POST` | `/predict/batch` | Score up to 500 customers at once |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Example: Single Customer Prediction

**Request**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 286.50
  }'
```

**Response**
```json
{
  "customer_id": null,
  "churn_probability": 0.8423,
  "churn_flag": 1,
  "risk_tier": "HIGH",
  "business_action": "HIGH RISK — Trigger immediate retention campaign",
  "latency_ms": 3.2
}
```

---

## 📈 Key EDA Insights

- **Contract type** is the #1 business predictor — month-to-month customers churn at **5× the rate** of two-year contract holders
- **Fiber optic** subscribers churn at nearly **3× the rate** of DSL customers
- **Churners cluster in the first 12 months** — early intervention is highest-ROI
- Customers paying **above-average monthly charges** are disproportionately represented in the churn group

---

## ⚙️ XGBoost Hyperparameters

```python
XGBClassifier(
    n_estimators    = 500,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.85,
    colsample_bytree= 0.85,
    min_child_weight= 3,
    gamma           = 0.2,
    reg_alpha       = 0.05,    # L1 regularisation
    reg_lambda      = 2.0,     # L2 regularisation
    eval_metric     = "logloss",
    random_state    = 42,
)
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost |
| API Framework | FastAPI + Uvicorn |
| Data | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Serialisation | joblib |

---

## 👤 Author

**Vaibhav Kose** — Data Analyst  
Built as a portfolio project to demonstrate end-to-end ML engineering skills.

---

## 📄 License

MIT License — free to use, modify, and distribute.
