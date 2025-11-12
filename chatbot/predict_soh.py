# ⚙️ predict_soh.py
"""
Predicts the State of Health (SOH) of a battery using a trained regression model.
Then classifies each prediction as either "Healthy" or "Has a Problem" based on SOH value.

Features:
- Loads serialized model (models/soh_linear_model.pkl)
- Accepts feature vector for U1–U21
- Returns numeric SOH prediction
- Applies threshold-based classification (default 0.6) -> 'Healthy' or 'Has a Problem'
- Allows custom threshold via parameter or variable input
"""

import os
import joblib
import pandas as pd
import numpy as np

# --- Step 1: Configuration: User-configurable threshold variable ---
# Default threshold for classification (can be modified here or passed as function argument)
DEFAULT_THRESHOLD = 0.6

# --- Step 2: Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Navigate up from /chatbot to project root

MODEL_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "models", "soh_linear_model.pkl"))
DATA_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "cleaned_pulsebat.csv"))
OUTPUT_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "results", "predicted_soh_with_labels.csv"))

print(f"DEBUG: MODEL_PATH = {MODEL_PATH}")
print(f"DEBUG: Checking if file exists: {os.path.exists(MODEL_PATH)}")

# --- Step 3: Classification Function ---
def classify_soh(value, threshold=None):
    """
    Classify a SOH value as 'Healthy' or 'Has a Problem' based on threshold.
    
    Args:
        value (float): SOH value to classify
        threshold (float, optional): Threshold value. If None, uses DEFAULT_THRESHOLD.
    
    Returns:
        str: 'Healthy' if value >= threshold, else 'Has a Problem'
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    return "Healthy" if value >= threshold else "Has a Problem"

# --- Step 4: Model Loading Function ---
def load_model(model_path=None):
    """
    Load the trained SOH prediction model.
    
    Args:
        model_path (str, optional): Path to the model file. If None, uses default MODEL_PATH.
    
    Returns:
        Trained model object
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

# --- Step 5: Batch Prediction Function ---
def predict_soh_batch(data_path=None, threshold=None, model_path=None, output_path=None):
    """
    Predict SOH for all samples in the dataset with configurable threshold.
    
    Args:
        data_path (str, optional): Path to CSV file with features. If None, uses default DATA_PATH.
        threshold (float, optional): Classification threshold. If None, uses DEFAULT_THRESHOLD.
        model_path (str, optional): Path to model file. If None, uses default MODEL_PATH.
        output_path (str, optional): Path to save results. If None, uses default OUTPUT_PATH.
    
    Returns:
        pd.DataFrame with 'Predicted_SOH' and 'Condition' columns
    
    Raises:
        FileNotFoundError: If model or data file doesn't exist
        ValueError: If features don't have exactly 21 columns
    """
    # Set default values
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if model_path is None:
        model_path = MODEL_PATH
    if data_path is None:
        data_path = DATA_PATH
    if output_path is None:
        output_path = OUTPUT_PATH
    
    # Load model
    model = load_model(model_path)
    print(f"Model loaded successfully from: {model_path}\n")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data loaded from: {data_path} (shape {df.shape})")
    
    # Select features (U1-U21)
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
    print(f"Using feature matrix of shape: {X_test.shape}")
    
    # Predict numeric SOH values
    y_pred = model.predict(X_test)
    
    # Classify with configurable threshold
    soh_status = np.array([classify_soh(v, threshold=threshold) for v in y_pred])
    
    # Combine predictions and classifications
    results = pd.DataFrame({
        "Predicted_SOH": y_pred,
        "Condition": soh_status
    })
    
    # Display summary
    print("=== Battery SOH Prediction Results (Preview) ===")
    print(results.head())
    
    counts = results["Condition"].value_counts()
    total = len(results)
    print("\n=== Condition Summary ===")
    for label, count in counts.items():
        print(f"{label}: {count} ({count / total * 100:.2f}%)")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Threshold used: {threshold}")
    
    return results

# --- Step 6: Single Prediction Function ---
def predict_soh(features, threshold=None, model=None):
    """
    Predict SOH for a single battery sample and classify the result.
    
    Args:
        features (list[float] or np.ndarray): Feature vector with 21 values (U1-U21)
        threshold (float, optional): Classification threshold. If None, uses DEFAULT_THRESHOLD.
        model: Trained model object. If None, loads model automatically.
    
    Returns:
        dict: Dictionary with keys:
            - 'soh' (float): Predicted SOH value
            - 'condition' (str): 'Healthy' or 'Has a Problem'
    
    Raises:
        ValueError: If features length is not 21
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    # Validate input length
    features_array = np.array(features)
    if len(features_array) != 21:
        raise ValueError(f"Expected 21 feature values (U1..U21), but got {len(features_array)}")
    
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Reshape for prediction (model expects 2D array)
    features_2d = features_array.reshape(1, -1)
    
    # Predict SOH
    soh_prediction = model.predict(features_2d)[0]
    
    # Classify
    condition = classify_soh(soh_prediction, threshold)
    
    return {
        'soh': float(soh_prediction),
        'condition': condition
    }

# --- Step 7: Main Execution (for script usage) ---
if __name__ == "__main__":
    # User-configurable threshold variable (can be modified here)
    # Modify this value to change the threshold for batch processing
    # Examples: 0.5, 0.6, 0.7, etc.
    THRESHOLD = DEFAULT_THRESHOLD  # Change this line to use a different threshold
    
    print(f"Using threshold: {THRESHOLD}")
    print("=" * 60)
    
    # Run prediction with configurable threshold
    results = predict_soh_batch(threshold=THRESHOLD)
