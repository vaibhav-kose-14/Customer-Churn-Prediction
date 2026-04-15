import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get current folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and data
model = joblib.load(os.path.join(script_dir, 'xgboost_production_model.pkl'))
X_test = np.load(os.path.join(script_dir, 'X_test.npy'))

# Load preprocessor to get ACTUAL feature names after encoding
preprocessor = joblib.load(os.path.join(script_dir, 'preprocessor_pipeline.pkl'))

# Get the real feature names after preprocessing (one-hot encoding expands columns)
try:
    feature_names = preprocessor.get_feature_names_out()
    print(f"Features after preprocessing: {len(feature_names)}")
except:
    # Fallback: generate generic names
    feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    print(f"Using generic feature names: {len(feature_names)}")

# Create DataFrame with correct feature names
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"Loaded {X_test.shape[0]} test samples with {X_test.shape[1]} features")

# SHAP explainer
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (top 10 features)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'shap_summary.png'), dpi=150)
print("✅ Saved: shap_summary.png")

# Feature importance bar chart
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, plot_type="bar", max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'shap_importance.png'), dpi=150)
print("✅ Saved: shap_importance.png")

print("\nTop 10 most important features for churn prediction:")
feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10))