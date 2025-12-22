"""
predict.py
-----------
Helper functions to load saved model artifacts and make predictions.
This module is used by the Flask app to serve predictions.
"""
import os
import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'model')


def load_artifacts():
    """Load and return (model, label_encoder, scaler).

    The model files are created by `train_model.py`.
    """
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    if not (os.path.exists(model_path) and os.path.exists(le_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError('Model artifacts not found. Run backend/train_model.py first to create them.')

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    scaler = joblib.load(scaler_path)

    return model, le, scaler


def predict_crop(features: list):
    """Given a list of 7 features in order:
       [Temperature, Humidity, Rainfall, Soil_pH, Nitrogen, Phosphorus, Potassium]
       Return a tuple: (crop_name, confidence_float)
    """
    model, le, scaler = load_artifacts()
    x = np.array(features, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    probs = model.predict_proba(x_scaled)
    top_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs))
    crop_name = le.inverse_transform([top_idx])[0]

    return crop_name, confidence
