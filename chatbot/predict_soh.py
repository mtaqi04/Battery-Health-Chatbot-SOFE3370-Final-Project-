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
