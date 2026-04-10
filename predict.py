"""
predict.py
==========
Demonstrates real-time single-customer churn scoring — the 'real-time
scoring' claim on the portfolio resume.

Loads:
  • preprocessor_pipeline.pkl      (ColumnTransformer fitted in preprocess.py)
  • xgboost_production_model.pkl   (XGBClassifier trained in train.py)

Accepts a mock JSON payload representing one customer, preprocesses it
through the exact same pipeline used at training time, and outputs:
  • Churn probability score  (0.00 – 1.00)
  • Risk tier label          (LOW / MEDIUM / HIGH)
  • Business action trigger  (retention campaign / alert / safe)

Usage:
  python predict.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH  = os.path.join(BASE_DIR, "preprocessor_pipeline.pkl")
MODEL_PATH     = os.path.join(BASE_DIR, "xgboost_production_model.pkl")
COLUMNS_PATH   = os.path.join(BASE_DIR, "feature_columns.pkl")

# ── Mock Customer Payloads ────────────────────────────────────────────────────
# Three customers with different churn risk profiles — mirrors what a
# downstream API / microservice would POST to a scoring endpoint.

CUSTOMERS = [
    {
        "_id": "CUST-7841",
        "_label": "Month-to-Month, Fiber, Short Tenure → HIGH RISK",
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
        "TotalCharges": 286.50,
    },
    {
        "_id": "CUST-3302",
        "_label": "One-Year Contract, Moderate Tenure → MEDIUM RISK",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 65.75,
        "TotalCharges": 1183.50,
    },
    {
        "_id": "CUST-1129",
        "_label": "Two-Year Contract, Long Tenure, Full Support → LOW RISK",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 58,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card",
        "MonthlyCharges": 54.25,
        "TotalCharges": 3146.50,
    },
]

# ── Risk Tier Logic ───────────────────────────────────────────────────────────
def classify_risk(prob: float) -> tuple[str, str]:
    """
    Returns (risk_label, action_message) based on churn probability.
    Thresholds are tunable business parameters — here:
      < 30%  → LOW    (monitor)
      30-50% → MEDIUM (proactive outreach)
      > 50%  → HIGH   (immediate retention campaign)
    """
    if prob > 0.50:
        return "HIGH",   "🚨  HIGH RISK: Triggering retention campaign immediately!"
    elif prob > 0.30:
        return "MEDIUM", "⚠️   MEDIUM RISK: Schedule proactive outreach within 7 days."
    else:
        return "LOW",    "✅  LOW RISK: Customer stable — no immediate action required."


# ── Single-Customer Scorer ────────────────────────────────────────────────────
def score_customer(payload: dict,
                   preprocessor,
                   model,
                   feature_columns: list) -> dict:
    """
    Accepts a raw customer JSON dict, runs it through the preprocessing
    pipeline, and returns a scoring response dict.
    """
    # Strip internal metadata keys
    features = {k: v for k, v in payload.items() if not k.startswith("_")}

    # Build a single-row DataFrame — column order must match training
    df_input = pd.DataFrame([features])[feature_columns]

    # Preprocess (imputation + scaling + encoding)
    X_transformed = preprocessor.transform(df_input)

    # Score
    churn_prob   = float(model.predict_proba(X_transformed)[0, 1])
    churn_flag   = int(churn_prob > 0.50)
    risk, action = classify_risk(churn_prob)

    return {
        "customer_id":       payload["_id"],
        "churn_probability": round(churn_prob, 4),
        "churn_flag":        churn_flag,          # 1 = likely to churn
        "risk_tier":         risk,
        "business_action":   action,
    }


# ── Pretty Printer ────────────────────────────────────────────────────────────
RISK_COLORS = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[92m"}
RESET = "\033[0m"

def print_result(payload: dict, result: dict):
    color = RISK_COLORS.get(result["risk_tier"], "")
    print(f"\n  {'─'*56}")
    print(f"  Customer ID  : {result['customer_id']}")
    print(f"  Profile      : {payload['_label']}")
    print(f"  {'─'*56}")
    print(f"  Churn Prob   : {result['churn_probability']:.2%}")
    print(f"  Risk Tier    : {color}{result['risk_tier']}{RESET}")
    print(f"  Churn Flag   : {'YES — will likely churn' if result['churn_flag'] else 'NO — likely to stay'}")
    print(f"\n  {color}{result['business_action']}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  CUSTOMER CHURN — REAL-TIME SCORING ENGINE")
    print("=" * 60)

    # -- Load artefacts -------------------------------------------------------
    print("\n[1/3] Loading serialised artefacts...")
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found: {PIPELINE_PATH}\n"
                                f"  → Run preprocess.py first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\n"
                                f"  → Run train.py first.")

    preprocessor    = joblib.load(PIPELINE_PATH)
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)

    print(f"  ✔  Preprocessor loaded  →  {PIPELINE_PATH}")
    print(f"  ✔  XGBoost model loaded →  {MODEL_PATH}")
    print(f"  ✔  Feature schema       →  {len(feature_columns)} columns")

    # -- Simulate receiving JSON payloads -------------------------------------
    print("\n[2/3] Simulating real-time scoring API requests...\n")
    print("  Scenario: Three customer records arrive as JSON payloads")
    print("  (e.g., from a web app, CRM event stream, or Kafka topic)")

    # -- Score each customer --------------------------------------------------
    print("\n[3/3] Scoring Results")
    print("=" * 60)

    all_results = []
    for payload in CUSTOMERS:
        t0 = time.time()
        result = score_customer(payload, preprocessor, model, feature_columns)
        latency_ms = (time.time() - t0) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        all_results.append(result)
        print_result(payload, result)
        print(f"\n  ⏱  Scoring latency: {latency_ms:.2f} ms  "
              f"(avg single-record inference)")

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SCORING SUMMARY")
    print("=" * 60)
    print(f"\n  {'Customer':<12} {'Churn Prob':>12} {'Risk':>8} {'Flag':>6}")
    print(f"  {'─'*46}")
    for r in all_results:
        color  = RISK_COLORS.get(r["risk_tier"], "")
        flag   = "CHURN" if r["churn_flag"] else "STAY"
        print(f"  {r['customer_id']:<12} {r['churn_probability']:>11.2%} "
              f"  {color}{r['risk_tier']:>6}{RESET}  {flag:>6}")

    avg_lat = np.mean([r["latency_ms"] for r in all_results])
    print(f"\n  Average scoring latency: {avg_lat:.2f} ms per customer")
    print("\n  ✔  Real-time scoring demonstration complete.\n")

    # -- JSON output (mirrors what an API endpoint would return) --------------
    print("  Raw JSON response (as returned by a REST scoring endpoint):")
    print("  " + "─" * 56)
    clean = [{k: v for k, v in r.items() if k != "business_action"}
             for r in all_results]
    print("  " + json.dumps(clean, indent=4).replace("\n", "\n  "))
    print()


if __name__ == "__main__":
    main()
