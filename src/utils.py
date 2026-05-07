# src/utils.py
import joblib
import pandas as pd
import numpy as np
import os
from config import (
    BASE_FEATURES, EXTERNAL_FEATURES, HYBRID_ALPHA, HYBRID_BETA, 
    CONGESTION_WEIGHTS, RED_LINE, BLUE_LINE, GREEN_LINE
)

def load_artifacts():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    # Update these filenames if you named them differently in the notebook!
    model = joblib.load(os.path.join(base_path, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'feature_scaler.pkl'))
    station_encoder = joblib.load(os.path.join(base_path, 'station_id_encoder.pkl'))
    route_encoder = joblib.load(os.path.join(base_path, 'route_encoder.pkl'))
    return model, scaler, station_encoder, route_encoder

def load_station_mapping():
    stops_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'stops.txt'))
    if os.path.exists(stops_path):
        stops_df = pd.read_csv(stops_path)
        return dict(zip(stops_df['stop_id'], stops_df['stop_name']))
    return {}

def get_ui_station_mappings(station_encoder, station_mapping):
    valid_encoded_ids = list(station_encoder.classes_)
    unique_station_names = sorted(list(set([station_mapping.get(sid, sid) for sid in valid_encoded_ids])))
    name_to_raw_id = {}
    for sid in valid_encoded_ids:
        name = station_mapping.get(sid, sid)
        if name not in name_to_raw_id:
            name_to_raw_id[name] = sid
    return unique_station_names, name_to_raw_id

def normalize(name):
    return name.strip().lower()

def get_route_from_station(station_name):
    norm_name = normalize(station_name)
    if norm_name in [normalize(s) for s in RED_LINE]: return 'Red'
    if norm_name in [normalize(s) for s in BLUE_LINE]: return 'Blue'
    if norm_name in [normalize(s) for s in GREEN_LINE]: return 'Green'
    return 'Unknown'

def get_stop_sequence(station_name, route_name):
    norm_name = normalize(station_name)
    line = []
    if route_name == 'Red': line = [normalize(s) for s in RED_LINE]
    elif route_name == 'Blue': line = [normalize(s) for s in BLUE_LINE]
    elif route_name == 'Green': line = [normalize(s) for s in GREEN_LINE]
    
    if norm_name in line: return line.index(norm_name) + 1
    return 10

def estimate_stops_between(source, dest):
    norm_source, norm_dest = normalize(source), normalize(dest)
    lines = [
        [normalize(s) for s in RED_LINE], 
        [normalize(s) for s in BLUE_LINE], 
        [normalize(s) for s in GREEN_LINE]
    ]
    
    for line in lines:
        if norm_source in line and norm_dest in line:
            return abs(line.index(norm_source) - line.index(norm_dest))
            
    interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground']
    best_dist = 999
    for interchange in interchanges:
        norm_interchange = normalize(interchange)
        dist1, dist2 = 999, 999
        for line in lines:
            if norm_source in line and norm_interchange in line:
                dist1 = abs(line.index(norm_source) - line.index(norm_interchange))
            if norm_dest in line and norm_interchange in line:
                dist2 = abs(line.index(norm_dest) - line.index(norm_interchange))
        if dist1 != 999 and dist2 != 999:
            best_dist = min(best_dist, dist1 + dist2)
            
    return best_dist if best_dist != 999 else 5

def get_dynamic_travel_time():
    # Scheduled travel time per stop in HYD Metro is roughly 2.5 minutes
    # The static timetable does not change; delays handle the peak/weekend variations.
    return 2.5

# --- NEW HYBRID LOGIC BELOW ---

def calculate_oas(row):
    """Calculates the physical Operational Adjustment Score"""
    # Is it a perfectly clear day with no events?
    is_normal_day = (row['weather_severity'] == 0.0) and (row['event_intensity'] == 0.0)
    
    if is_normal_day:
        base_congestion_penalty = 0.0  # Zero physics penalty on normal days
    else:
        # Only apply congestion penalty if the network is already stressed
        base_congestion_penalty = max(0, (row['station_congestion'] - 1.0) * 0.2)
        
    score = (row['weather_severity'] * 0.8) + \
            (row['event_intensity'] * 2.5) + \
            base_congestion_penalty
    return score

def estimate_total_delay(source_name, dest_name, hour, day_int, station_encoder, route_encoder, name_to_raw_id, weather_sev, event_int, model, scaler, initial_delay=0.0):
    num_stops = estimate_stops_between(source_name, dest_name)
    
    source_line = get_route_from_station(source_name)
    dest_line = get_route_from_station(dest_name)
    
    # 5-min buffer only if changing lines
    interchange_penalty = 5.0 if source_line != dest_line else 0.0
    base_travel_time = (num_stops * get_dynamic_travel_time()) + interchange_penalty
    
    current_delay = initial_delay
    final_trip_delay = 0.0
    
    for i in range(num_stops):
        simulated_station = source_name if i == 0 else "Transit_Stop"
        raw_id = name_to_raw_id.get(simulated_station, 'Unknown')
        
        features_df = derive_features(
            raw_id, hour, day_int, current_delay, 
            station_encoder, route_encoder, simulated_station,
            weather_sev, event_int
        )
        
        step_delay, _, _ = make_prediction(features_df, model, scaler)
        
        # --- UPGRADED PROPAGATION LOGIC ---
        if weather_sev == 0.0 and event_int == 0.0:
            # On a clear day, the delay is crushed and absorbed by the buffer
            current_delay = step_delay * 0.55
        else:
            # During storms/events, delays propagate severely
            current_delay = step_delay * 0.15 
            
        final_trip_delay = step_delay
        
    return final_trip_delay, base_travel_time, num_stops

def derive_features(station_id, hour, day_of_week_int, prev_delay, station_encoder, route_encoder, 
                    station_name, weather_sev=0.0, event_int=0.0):
    
    try: station_id_encoded = station_encoder.transform([station_id])[0]
    except: station_id_encoded = 0
        
    is_peak = 1 if (8 <= hour <= 10) or (17 <= hour <= 20) else 0
    is_weekend = 1 if day_of_week_int >= 5 else 0
    station_congestion = CONGESTION_WEIGHTS.get(station_name, 1.0)
    route_name = get_route_from_station(station_name)
    
    try: route_encoded = route_encoder.transform([route_name])[0]
    except: route_encoded = 0
        
    stop_sequence = get_stop_sequence(station_name, route_name)
    
    # --- UPGRADED RECOVERY PHYSICS ---
    recovery = 0.0
    if stop_sequence > 18: recovery += 0.2
    if is_peak == 0: recovery += 0.2
    
    # HYPER-BOOST: If the weather is clear and no events, the train easily catches up to schedule
    if weather_sev == 0.0 and event_int == 0.0:
        recovery += 0.6  # Massive 60% recovery boost
        
    recovery = min(recovery, 0.95) # Cap at 95% to prevent negative delays

    features = {
        'station_id_encoded': station_id_encoded,
        'hour': hour,
        'is_peak': is_peak,
        'station_congestion': station_congestion,
        'stop_sequence': stop_sequence,
        'prev_delay': prev_delay,
        'route_encoded': route_encoded,
        'day_of_week': day_of_week_int,
        'is_weekend': is_weekend,
        'weather_severity': weather_sev,
        'event_intensity': event_int,
        'recovery_factor': recovery
    }
    return pd.DataFrame([features])

def make_prediction(features_df, model, scaler):
    """The Hybrid Blending Predictor"""
    # 1. Base ML Prediction (Schedule Data Only)
    X_base = features_df[BASE_FEATURES]
    X_base_scaled = scaler.transform(X_base)
    xgb_pred = max(0, model.predict(X_base_scaled)[0])
    
    # 2. OAS Prediction (External Physics)
    oas_score = calculate_oas(features_df.iloc[0])
    
    # 3. Hybrid Blender & Recovery
    recovery = features_df.iloc[0]['recovery_factor']
    raw_hybrid = (HYBRID_ALPHA * xgb_pred) + (HYBRID_BETA * oas_score)
    final_delay = max(0, raw_hybrid * (1 - recovery))
    
    return final_delay, xgb_pred, oas_score

def get_contextual_feedback(final_delay, oas_score, is_peak, weather_sev):
    if final_delay < 2.0:
        return "🟢 Good Service: The network is operating smoothly. Minor variations only."
    elif final_delay < 5.0:
        msg = "🟡 Minor Delays Expected. "
        if is_peak: msg += "Standard peak-hour congestion is slowing operations."
        if weather_sev > 0: msg += "Weather conditions are slightly reducing speed limits."
        return msg
    else:
        msg = "🔴 Significant Disruptions: "
        if oas_score > 3.0: 
            msg += "External environmental factors (Weather/Events) are severely impacting schedule adherence."
        else:
            msg += "Network bottlenecks and delay propagation are highly active."
        return msg