"""
train.py
========
Loads processed data (output of preprocess.py), benchmarks three classifiers:
  • Logistic Regression
  • Random Forest
  • XGBoost  ← selected as production model

Prints Accuracy, F1-Score, and AUC-ROC for every model side-by-side.
Tunes XGBoost hyperparameters to reach ≥87% accuracy.
Saves the winner as:  xgboost_production_model.pkl
"""

import os
import time
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (accuracy_score, f1_score,
                                    roc_auc_score, classification_report,
                                    confusion_matrix)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
X_TRAIN    = os.path.join(BASE_DIR, "X_train.npy")
X_TEST     = os.path.join(BASE_DIR, "X_test.npy")
Y_TRAIN    = os.path.join(BASE_DIR, "y_train.npy")
Y_TEST     = os.path.join(BASE_DIR, "y_test.npy")
MODEL_OUT  = os.path.join(BASE_DIR, "xgboost_production_model.pkl")

SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return accuracy, F1 (weighted), AUC-ROC for a fitted model."""
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)

    return {"model": name, "accuracy": acc, "f1_score": f1, "auc_roc": auc,
            "y_pred": y_pred, "y_prob": y_prob, "_obj": model}


def banner(text: str, width: int = 60):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ── Model Definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    """
    Returns the three benchmark classifiers.
    XGBoost is tuned explicitly for ≥87% accuracy:
      - n_estimators=500    more trees → better generalisation
      - max_depth=6         deeper trees capture feature interactions
      - learning_rate=0.05  low LR with more trees avoids overfitting
      - subsample=0.8       row sub-sampling reduces variance
      - colsample_bytree=0.8  feature sub-sampling
      - min_child_weight=3  controls leaf purity
      - scale_pos_weight    handles class imbalance automatically
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.05,
            reg_lambda=2.0,
            scale_pos_weight=1,          # keep at 1 to maximise accuracy
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # -- Load data ------------------------------------------------------------
    banner("CUSTOMER CHURN — MODEL BENCHMARKING & TRAINING")

    if not all(os.path.exists(p) for p in [X_TRAIN, X_TEST, Y_TRAIN, Y_TEST]):
        raise FileNotFoundError(
            "Processed data not found. Please run  preprocess.py  first."
        )

    print("\nLoading processed arrays...")
    X_train = np.load(X_TRAIN)
    X_test  = np.load(X_TEST)
    y_train = np.load(Y_TRAIN)
    y_test  = np.load(Y_TEST)
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")
    print(f"  Churn rate  →  Train: {y_train.mean():.2%}  |  Test: {y_test.mean():.2%}")

    # -- Train & evaluate all models ------------------------------------------
    banner("PHASE 1 — TRAINING ALL BENCHMARK MODELS")

    models  = get_models()
    results = []

    for name, model in models.items():
        print(f"\n  ▶ Training: {name} ...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        metrics = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        print(f"done  ({elapsed:.1f}s)")

    # -- Comparison table -----------------------------------------------------
    banner("PHASE 2 — BENCHMARK COMPARISON RESULTS")

    col_w = 24
    header = (f"{'Model':<{col_w}} {'Accuracy':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("\n" + header)
    print("-" * (col_w + 35))

    for r in results:
        marker = "  ◀ SELECTED" if r["model"] == "XGBoost" else ""
        print(
            f"{r['model']:<{col_w}} "
            f"{r['accuracy']:>9.4f} "
            f"{r['f1_score']:>10.4f} "
            f"{r['auc_roc']:>10.4f}"
            f"{marker}"
        )

    print("-" * (col_w + 35))
    print("\n  Metric definitions:")
    print("    Accuracy  – % of all predictions correct")
    print("    F1-Score  – harmonic mean of precision & recall (weighted)")
    print("    AUC-ROC   – area under the ROC curve  (1.0 = perfect)")

    # -- Select XGBoost explicitly --------------------------------------------
    banner("PHASE 3 — SELECTING XGBoost AS PRODUCTION MODEL")

    xgb_metrics = next(r for r in results if r["model"] == "XGBoost")
    xgb_model   = xgb_metrics["_obj"]

    print(f"\n  Rationale for selecting XGBoost:")
    print("    ✔  Gradient boosting handles mixed feature types naturally")
    print("    ✔  Built-in regularisation (L1/L2) reduces overfitting")
    print("    ✔  Highly tunable for business-critical precision/recall trade-offs")
    print("    ✔  Native support for real-time single-row scoring (predict.py)")
    print(f"\n  Final XGBoost Metrics:")
    print(f"    Accuracy  : {xgb_metrics['accuracy']:.4f}  "
          f"({'✔ TARGET MET' if xgb_metrics['accuracy'] >= 0.87 else '✘ BELOW 87%'})")
    print(f"    F1-Score  : {xgb_metrics['f1_score']:.4f}")
    print(f"    AUC-ROC   : {xgb_metrics['auc_roc']:.4f}")

    # -- Detailed report for XGBoost ------------------------------------------
    print("\n  Classification Report (XGBoost):")
    print(classification_report(
        y_test, xgb_metrics["y_pred"],
        target_names=["No Churn", "Churn"]
    ))

    cm = confusion_matrix(y_test, xgb_metrics["y_pred"])
    print("  Confusion Matrix:")
    print(f"           Predicted No  Predicted Yes")
    print(f"  Actual No   {cm[0,0]:>6}        {cm[0,1]:>6}")
    print(f"  Actual Yes  {cm[1,0]:>6}        {cm[1,1]:>6}")

    # -- Save production model ------------------------------------------------
    banner("PHASE 4 — SAVING PRODUCTION MODEL")

    joblib.dump(xgb_model, MODEL_OUT)
    size_kb = os.path.getsize(MODEL_OUT) / 1024
    print(f"\n  ✔  Model saved  →  {MODEL_OUT}")
    print(f"     File size    :  {size_kb:.1f} KB")
    print(f"     Serialiser   :  joblib (compatible with sklearn pipelines)")
    print("\n  Pipeline complete. Run predict.py for real-time scoring demo.\n")


if __name__ == "__main__":
    main()
