# src/utils.py
import joblib
import pandas as pd
import os
from config import FEATURE_COLUMNS, CONGESTION_WEIGHTS

def load_artifacts():
    """Loads the trained model, scaler, and encoders."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    try:
        model = joblib.load(os.path.join(base_path, 'xgboost.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'feature_scaler.pkl'))
        station_encoder = joblib.load(os.path.join(base_path, 'station_id_encoder.pkl'))
        route_encoder = joblib.load(os.path.join(base_path, 'route_encoder.pkl'))
        return model, scaler, station_encoder, route_encoder
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model artifact: {e}. Ensure all .pkl files are in the models/ folder.")

def derive_features(hour, day_value, station_name):
    """Computes automated features based on raw inputs."""
    is_peak = 1 if (8 <= hour <= 10) or (17 <= hour <= 20) else 0
    is_weekend = 1 if day_value >= 5 else 0
    station_congestion = CONGESTION_WEIGHTS.get(station_name, 1.0) # Default to 1.0
    
    return is_peak, is_weekend, station_congestion

def make_prediction(model, scaler, input_data):
    """Formats data, applies scaling, and returns the prediction."""
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]
    return max(0, round(prediction, 1)) # Prevent negative delays

def get_contextual_feedback(delay, is_peak, prev_delay):
    """Generates UI color and explanation based on the prediction."""
    # Determine color and status
    if delay <= 2.0:
        color = "green"
        status = "On Time / Minor Delay"
    elif delay <= 7.0:
        color = "orange"
        status = "Moderate Delay"
    else:
        color = "red"
        status = "Severe Delay"
        
    # Generate simple explanation
    reasons = []
    if prev_delay > 2.0:
        reasons.append("cascading delays from the previous station")
    if is_peak == 1:
        reasons.append("heavy peak hour traffic")
    
    explanation = "System running smoothly."
    if reasons:
        explanation = f"Higher delay expected due to {' and '.join(reasons)}."
        
    return color, status, explanation