# ⚙️ predict_soh.py (placeholder instructions)
"""
Purpose:
- Load serialized model (models/soh_linear_model.pkl)
- Accept feature vector for U1–U21
- Return numeric SOH prediction
- Apply threshold (default 0.6) -> 'Healthy' or 'Has a Problem'
- Allow custom threshold via parameter or UI control

TODO:
- Implement load_model() using joblib
- Implement predict_soh(features: list[float], threshold: float = 0.6)
- Validate input length == 21
- Add simple tests if using pytest
"""
# ⚙️ predict_soh.py (placeholder instructions)
"""
Purpose:
- Load serialized model (models/soh_linear_model.pkl)
- Accept feature vector for U1–U21
- Return numeric SOH prediction
- Apply threshold (default 0.6) -> 'Healthy' or 'Has a Problem'
- Allow custom threshold via parameter or UI control

TODO:
- Implement load_model() using joblib
- Implement predict_soh(features: list[float], threshold: float = 0.6)
- Validate input length == 21
- Add simple tests if using pytest
"""

"""
predict_soh.py
-----------------------
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.
"""

"""
predict_soh.py
-----------------------
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.
"""
"""
predict_soh.py
--------------------------------
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.
"""

"""
predict_soh.py
---------------------------------------
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.
"""
"""
predict_soh.py
-----------------------
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.
"""
import os
import joblib
import pandas as pd
import numpy as np

# --- Step 1: Compute absolute path relative to this script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "soh_linear_model.pkl")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "cleaned_pulsebat.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "results", "predicted_soh_with_labels.csv")

# --- Step 2: Load trained model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded successfully from: {MODEL_PATH}\n")

# --- Step 3: Load test dataset and select features ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded from: {DATA_PATH} (shape {df.shape})")

# Ensure we use exactly the 21 features the model expects
feature_cols = [f"U{i}" for i in range(1, 22)]
if set(feature_cols).issubset(df.columns):
    X_test_df = df[feature_cols]
else:
    # Remove common non-feature columns if present (target, unnamed index)
    drop_cols = []
    if "SOH" in df.columns:
        drop_cols.append("SOH")
    drop_cols += [c for c in df.columns if str(c).startswith("Unnamed")]
    X_test_df = df.drop(columns=drop_cols, errors='ignore')
    print(f"Note: selected columns after dropping {drop_cols}: {list(X_test_df.columns)[:5]}...")

# Validate feature count
if X_test_df.shape[1] != 21:
    raise ValueError(f"Expected 21 feature columns (U1..U21), but got {X_test_df.shape[1]} columns: {list(X_test_df.columns)}")

X_test = X_test_df.values
print(f"✅ Using feature matrix of shape: {X_test.shape}")

# --- Step 4: Predict numeric SOH values ---
y_pred = model.predict(X_test)

# --- Step 5: Classification rule ---
def classify_soh(value, threshold=0.6):
    return "Healthy" if value >= threshold else "Has a Problem"

soh_status = np.array([classify_soh(v) for v in y_pred])

# --- Step 6: Combine predictions and classifications ---
results = pd.DataFrame({
    "Predicted_SOH": y_pred,
    "Condition": soh_status
})

# --- Step 7: Display summary ---
print("=== 🔋 Battery SOH Prediction Results (Preview) ===")
print(results.head())

counts = results["Condition"].value_counts()
total = len(results)
print("\n=== 📊 Condition Summary ===")
for label, count in counts.items():
    print(f"{label}: {count} ({count / total * 100:.2f}%)")

# --- Step 8: Save results ---
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Results saved to: {OUTPUT_PATH}")
