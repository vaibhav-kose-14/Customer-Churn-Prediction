"""
app.py — Customer Churn Prediction API
=======================================
A production-style REST API built with FastAPI that wraps the trained
XGBoost churn model and preprocessing pipeline.

Endpoints
─────────
  GET  /              → welcome message + API info
  GET  /health        → model health check & metadata
  POST /predict       → score a single customer
  POST /predict/batch → score multiple customers at once

Swagger UI (interactive docs) auto-generated at:
  http://127.0.0.1:8000/docs

Run with:
  uvicorn app:app --reload --port 8000
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("churn_api")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH  = os.path.join(BASE_DIR, "preprocessor_pipeline.pkl")
MODEL_PATH     = os.path.join(BASE_DIR, "xgboost_production_model.pkl")
COLUMNS_PATH   = os.path.join(BASE_DIR, "feature_columns.pkl")

# ── Global model store (loaded once at startup) ───────────────────────────────
class ModelStore:
    preprocessor    = None
    model           = None
    feature_columns = None
    loaded_at       = None

store = ModelStore()


# ── Lifespan: load models on startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading model artefacts …")
    for path, label in [
        (PIPELINE_PATH, "preprocessor_pipeline.pkl"),
        (MODEL_PATH,    "xgboost_production_model.pkl"),
        (COLUMNS_PATH,  "feature_columns.pkl"),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(
                f"Artefact not found: {label}\n"
                f"  → Run preprocess.py then train.py first."
            )
    store.preprocessor    = joblib.load(PIPELINE_PATH)
    store.model           = joblib.load(MODEL_PATH)
    store.feature_columns = joblib.load(COLUMNS_PATH)
    store.loaded_at       = time.strftime("%Y-%m-%d %H:%M:%S")
    log.info("✔  All artefacts loaded successfully.")
    log.info("✔  Swagger docs → http://127.0.0.1:8000/docs")
    yield
    log.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Real-time churn scoring API for the Customer Churn Prediction portfolio project.\n\n"
        "**Pipeline:** Raw customer JSON → ColumnTransformer → XGBoost → Churn probability\n\n"
        "**Author:** Vaibhav Kose · Data Analyst Portfolio"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class CustomerInput(BaseModel):
    """
    All features required to score a single customer.
    Matches the schema produced by preprocess.py exactly.
    """
    # Demographics
    gender:           str   = Field(...,  example="Male",            description="Male or Female")
    SeniorCitizen:    int   = Field(...,  example=0,                 description="1 if senior citizen, else 0")
    Partner:          str   = Field(...,  example="No",              description="Yes / No")
    Dependents:       str   = Field(...,  example="No",              description="Yes / No")
    tenure:           float = Field(...,  example=3.0,               description="Months with the company")

    # Services
    PhoneService:     str   = Field(...,  example="Yes",             description="Yes / No")
    MultipleLines:    str   = Field(...,  example="No",              description="Yes / No / No phone service")
    InternetService:  str   = Field(...,  example="Fiber optic",     description="DSL / Fiber optic / No")
    OnlineSecurity:   str   = Field(...,  example="No",              description="Yes / No / No internet service")
    TechSupport:      str   = Field(...,  example="No",              description="Yes / No / No internet service")

    # Billing
    Contract:         str   = Field(...,  example="Month-to-month",  description="Month-to-month / One year / Two year")
    PaperlessBilling: str   = Field(...,  example="Yes",             description="Yes / No")
    PaymentMethod:    str   = Field(...,  example="Electronic check",description="Electronic check / Mailed check / Bank transfer / Credit card")
    MonthlyCharges:   float = Field(...,  example=95.50,             description="Monthly bill in USD")
    TotalCharges:     float = Field(...,  example=286.50,            description="Total billed to date in USD")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        valid = {"DSL", "Fiber optic", "No"}
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v

    @field_validator("tenure", "MonthlyCharges", "TotalCharges")
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Numeric fields must be ≥ 0")
        return v


class PredictionResponse(BaseModel):
    customer_id:       Optional[str] = Field(None, description="Optional customer identifier")
    churn_probability: float         = Field(...,  description="Probability of churn (0.00–1.00)")
    churn_flag:        int            = Field(...,  description="1 = will likely churn, 0 = likely to stay")
    risk_tier:         str            = Field(...,  description="LOW / MEDIUM / HIGH")
    business_action:   str            = Field(...,  description="Recommended retention action")
    latency_ms:        float          = Field(...,  description="Inference latency in milliseconds")


class BatchRequest(BaseModel):
    customers: list[CustomerInput] = Field(..., description="List of customer records (max 500)")

    @field_validator("customers")
    @classmethod
    def check_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("At least one customer is required")
        if len(v) > 500:
            raise ValueError("Batch size cannot exceed 500")
        return v


class BatchCustomerInput(CustomerInput):
    customer_id: Optional[str] = Field(None, example="CUST-7841",
                                        description="Optional ID to trace results back to source records")


class BatchRequest(BaseModel):
    customers: list[BatchCustomerInput] = Field(..., description="List of customer records (max 500)")


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(raw: dict) -> dict:
    """
    Replicates the interaction features built in preprocess.py.
    Must be called BEFORE passing data to the ColumnTransformer.
    """
    is_mtm  = 1.0 if raw["Contract"] == "Month-to-month" else 0.0
    has_sec = 1.0 if raw["OnlineSecurity"] == "Yes" else 0.0

    raw["MonthlyCharges_x_MTM"]  = raw["MonthlyCharges"] * is_mtm
    raw["Tenure_x_Security"]     = raw["tenure"] * has_sec
    raw["MTM_x_NoSecurity"]      = is_mtm * (1.0 - has_sec)
    raw["CostPerTenureMonth"]     = raw["MonthlyCharges"] / (raw["tenure"] + 1)
    return raw


def classify_risk(prob: float) -> tuple[str, str]:
    if prob > 0.50:
        return "HIGH",   "HIGH RISK — Trigger immediate retention campaign"
    elif prob > 0.30:
        return "MEDIUM", "MEDIUM RISK — Schedule proactive outreach within 7 days"
    else:
        return "LOW",    "LOW RISK — Customer stable, no immediate action required"


def score_one(customer: CustomerInput,
              customer_id: Optional[str] = None) -> dict:
    t0   = time.perf_counter()
    raw  = engineer_features(customer.model_dump())

    # Drop id field if present (from batch requests)
    raw.pop("customer_id", None)

    df   = pd.DataFrame([raw])[store.feature_columns]
    X    = store.preprocessor.transform(df)
    prob = float(store.model.predict_proba(X)[0, 1])
    flag = int(prob > 0.50)
    risk, action = classify_risk(prob)
    ms   = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "customer_id":       customer_id,
        "churn_probability": round(prob, 4),
        "churn_flag":        flag,
        "risk_tier":         risk,
        "business_action":   action,
        "latency_ms":        ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    return {
        "project":     "Customer Churn Prediction",
        "author":      "Vaibhav Kose",
        "version":     "1.0.0",
        "model":       "XGBoost (tuned, 87%+ accuracy)",
        "endpoints": {
            "health":        "GET  /health",
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch",
            "docs":          "GET  /docs",
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    """
    Health check endpoint.
    Returns model metadata and confirms the pipeline is loaded.
    Useful for deployment monitoring (Kubernetes liveness probe, etc.).
    """
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":           "healthy",
        "model":            "XGBoost",
        "n_features":       len(store.feature_columns),
        "loaded_at":        store.loaded_at,
        "pipeline_stages":  ["ColumnTransformer (impute + scale + encode)",
                             "XGBClassifier (n_estimators=500, max_depth=5)"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
async def predict(customer: CustomerInput, customer_id: Optional[str] = None):
    """
    **Score a single customer in real-time.**

    Pass a JSON payload with all customer features.
    Returns:
    - `churn_probability` — model confidence (0.00 – 1.00)
    - `risk_tier`         — LOW / MEDIUM / HIGH
    - `business_action`   — recommended retention action
    - `latency_ms`        — end-to-end inference time

    **Example high-risk customer:**
    ```json
    {
      "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
      "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
      "MultipleLines": "No", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "TechSupport": "No",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 95.50, "TotalCharges": 286.50
    }
    ```
    """
    try:
        result = score_one(customer, customer_id)
        log.info(f"Scored customer {customer_id or 'anon'} → "
                 f"prob={result['churn_probability']:.3f} "
                 f"risk={result['risk_tier']} "
                 f"({result['latency_ms']}ms)")
        return result
    except Exception as e:
        log.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Scoring"])
async def predict_batch(request: BatchRequest):
    """
    **Score a batch of customers (up to 500).**

    Accepts a list of customer records and returns a scored result
    for each one, plus a summary breakdown of risk tiers.

    Useful for:
    - Nightly scoring of all active customers
    - CRM export ingestion
    - Campaign targeting pipelines
    """
    t_batch = time.perf_counter()
    results = []

    try:
        for c in request.customers:
            cid    = getattr(c, "customer_id", None)
            result = score_one(c, cid)
            results.append(result)
    except Exception as e:
        log.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    total_ms   = round((time.perf_counter() - t_batch) * 1000, 2)
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        risk_counts[r["risk_tier"]] += 1

    log.info(f"Batch scored {len(results)} customers in {total_ms}ms "
             f"| HIGH={risk_counts['HIGH']} "
             f"MEDIUM={risk_counts['MEDIUM']} "
             f"LOW={risk_counts['LOW']}")

    return {
        "total_customers":  len(results),
        "batch_latency_ms": total_ms,
        "risk_summary":     risk_counts,
        "results":          results,
    }
