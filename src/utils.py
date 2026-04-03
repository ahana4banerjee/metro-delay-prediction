# src/utils.py
import joblib
import pandas as pd
import os
from config import FEATURE_COLUMNS, CONGESTION_WEIGHTS, RED_LINE, BLUE_LINE, GREEN_LINE

def load_artifacts():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model = joblib.load(os.path.join(base_path, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'phase3_feature_scaler.pkl'))
    station_encoder = joblib.load(os.path.join(base_path, 'station_id_encoder.pkl'))
    route_encoder = joblib.load(os.path.join(base_path, 'route_encoder.pkl'))
    return model, scaler, station_encoder, route_encoder

def load_station_mapping():
    """Reads stops.txt to map raw stop_ids to human-readable stop_names for the UI."""
    stops_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'stops.txt'))
    if os.path.exists(stops_path):
        stops_df = pd.read_csv(stops_path)
        # Create a dictionary mapping like {'MYP': 'Miyapur', 'AME': 'Ameerpet'}
        return dict(zip(stops_df['stop_id'], stops_df['stop_name']))
    return {} # Fallback if file isn't found

def derive_features(hour, day_value, station_name):
    is_peak = 1 if (8 <= hour <= 10) or (17 <= hour <= 20) else 0
    is_weekend = 1 if day_value >= 5 else 0
    station_congestion = CONGESTION_WEIGHTS.get(station_name, 1.0)
    return is_peak, is_weekend, station_congestion

def make_prediction(model, scaler, input_data):
    df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]
    return max(0, round(prediction, 1))

def get_contextual_feedback(delay, is_peak, prev_delay):
    if delay <= 2.0:
        return "green", "On Time / Minor Delay", "System running smoothly."
    elif delay <= 7.0:
        return "orange", "Moderate Delay", "Moderate traffic detected."
    
    reasons = []
    if prev_delay > 2.0: reasons.append("cascading delays")
    if is_peak == 1: reasons.append("peak hour traffic")
    explanation = f"Higher delay expected due to {' and '.join(reasons)}." if reasons else "Unplanned delay detected."
    return "red", "Severe Delay", explanation

def get_ui_station_mappings(station_encoder, station_mapping):
    """Deduplicates station names for the UI and creates a reverse lookup dictionary."""
    unique_names = []
    name_to_raw_id = {}
    
    for raw_id in station_encoder.classes_:
        name = station_mapping.get(raw_id, raw_id)
        # Only add the name if we haven't seen it yet (removes platform duplicates)
        if name not in name_to_raw_id:
            name_to_raw_id[name] = raw_id
            unique_names.append(name)
            
    return sorted(unique_names), name_to_raw_id

def get_route_from_station(station_name, available_routes):
    stat_upper = str(station_name).upper()
    if "NAGOLE" in stat_upper or "RAIDURG" in stat_upper:
        return next((r for r in available_routes if 'BLUE' in r.upper()), available_routes[0])
    elif "PARADE" in stat_upper or "JBS" in stat_upper:
        return next((r for r in available_routes if 'GREEN' in r.upper()), available_routes[0])
    else:
        return next((r for r in available_routes if 'RED' in r.upper()), available_routes[0])


def get_stop_sequence(station_name, route_name):
    """Automatically finds the stop sequence number for a given station."""
    route_upper = str(route_name).upper()
    
    try:
        if 'RED' in route_upper and station_name in RED_LINE:
            return RED_LINE.index(station_name) + 1
        elif 'BLUE' in route_upper and station_name in BLUE_LINE:
            return BLUE_LINE.index(station_name) + 1
        elif 'GREEN' in route_upper and station_name in GREEN_LINE:
            return GREEN_LINE.index(station_name) + 1
    except ValueError:
        pass
        
    return 5 # Safe fallback if something goes wrong

def estimate_stops_between(source_name, dest_name):
    """Calculates exact stops using real metro line topography."""
    lines = [RED_LINE, BLUE_LINE, GREEN_LINE]
    
    # 1. Check if they are on the same line
    for line in lines:
        if source_name in line and dest_name in line:
            return abs(line.index(source_name) - line.index(dest_name))
            
    # 2. If different lines, route through a major interchange
    interchanges = ['Ameerpet', 'MG Bus Station', 'Parade Ground']
    best_dist = 999
    
    for interchange in interchanges:
        dist1 = 999
        dist2 = 999
        for line in lines:
            if source_name in line and interchange in line:
                dist1 = abs(line.index(source_name) - line.index(interchange))
            if dest_name in line and interchange in line:
                dist2 = abs(line.index(dest_name) - line.index(interchange))
        
        # If a valid path through this interchange exists, check if it's the shortest
        if dist1 != 999 and dist2 != 999:
            best_dist = min(best_dist, dist1 + dist2)
            
    # Fallback just in case a name isn't mapped perfectly
    return best_dist if best_dist != 999 else 5

def get_dynamic_travel_time(is_peak, is_weekend):
    base_time = 2.5 
    if is_peak: base_time += 1.0   
    if is_weekend: base_time -= 0.5 
    return base_time

def get_dynamic_propagation(is_peak, station_congestion):
    base_prop = 0.1 
    prop = base_prop * station_congestion 
    if is_peak: prop *= 1.5               
    return min(0.5, prop)                 

def estimate_total_delay(current_delay, stops, is_peak, station_congestion):
    prop_factor = get_dynamic_propagation(is_peak, station_congestion)
    compounded = current_delay + (stops * prop_factor)
    return round(compounded, 1)