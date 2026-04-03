# src/config.py

# The exact order of features your XGBoost model expects
FEATURE_COLUMNS = [
    'station_id_encoded', 'hour', 'is_peak', 'station_congestion', 
    'stop_sequence', 'prev_delay', 'route_encoded', 
    'day_of_week', 'is_weekend'
]

# Station congestion weights (from Phase 2)
CONGESTION_WEIGHTS = {
    'Ameerpet': 3.5,
    'MG Bus Station': 3.0,
    'Parade Ground': 2.5,
    'Raidurg': 2.0, 
    'Secunderabad East': 2.0
}

# Standard days of the week mapping
DAYS_OF_WEEK = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}