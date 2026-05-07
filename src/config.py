# src/config.py

# ==========================================
# RESEARCH FRAMEWORK FEATURES (DECOUPLED)
# ==========================================
# 1. Base features that the XGBoost model was trained on
BASE_FEATURES = [
    'station_id_encoded', 'hour', 'is_peak', 'stop_sequence', 
    'prev_delay', 'route_encoded', 'day_of_week', 'is_weekend'
]

# 2. External features used exclusively by the Operational Adjustment Score (OAS)
EXTERNAL_FEATURES = [
    'weather_severity', 'event_intensity', 'station_congestion', 'recovery_factor'
]

# 3. Hybrid Blending Weights (Determined via optimization in notebook)
HYBRID_ALPHA = 0.95  # Weight of the XGBoost ML Model
HYBRID_BETA = 1.10   # Weight of the Operational Adjustment Score

# ==========================================
# UI MAPPINGS FOR ENVIRONMENTAL FACTORS
# ==========================================
WEATHER_CONDITIONS = {
    'Clear / Normal': 0.0,
    'Light Rain / Drizzle': 1.0,
    'Heavy Rain / Storm': 2.0,
    'Extreme Weather (Black Swan)': 3.5
}

EVENT_INTENSITY = {
    'No Event (Standard Ops)': 0.0,
    'Minor Local Event': 0.3,
    'Medium (Office Rush / Weekend Mall)': 0.6,
    'Major (IPL Match / Festival)': 1.0
}

# ==========================================
# EXISTING CONSTANTS
# ==========================================
CONGESTION_WEIGHTS = {
    'Ameerpet': 3.5,
    'MG Bus Station': 3.0,
    'Parade Ground': 2.5,
    'Raidurg': 2.0, 
    'Secunderabad East': 2.0
}

DAYS_OF_WEEK = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

RED_LINE = [
    'Miyapur', 'JNTU College', 'KPHB Colony', 'Kukatpally', 'Balanagar', 
    'Moosapet', 'Bharat Nagar', 'Erragadda', 'ESI Hospital', 'SR Nagar', 
    'Ameerpet', 'Punjagutta', 'Irrum Manzil', 'Khairatabad', 'Lakdikapul', 
    'Assembly', 'Nampally', 'Gandhi Bhavan', 'Osmania Medical College', 
    'MG Bus Station', 'Malakpet', 'New Market', 'Musarambagh', 'Dilsukhnagar', 
    'Chaitanyapuri', 'Victoria Memorial', 'LB Nagar'
]

BLUE_LINE = [
    'Raidurg', 'Hitec City', 'Durgam Cheruvu', 'Madhapur', 'Peddamma Gudi', 
    'Jubilee Hills Check Post', 'Road No 5 Jubilee Hills', 'Yusufguda', 
    'Madura Nagar', 'Ameerpet', 'Begumpet', 'Prakash Nagar', 'Rasoolpura', 
    'Paradise', 'Parade Ground', 'Secunderabad East', 'Mettuguda', 
    'Tarnaka', 'Habsiguda', 'NGRI', 'Stadium', 'Uppal', 'Nagole'
]

GREEN_LINE = [
    'JBS Parade Ground', 'Secunderabad West', 'Gandhi Hospital', 
    'Musheerabad', 'RTC Cross Roads', 'Chikkadpally', 'Narayanguda', 
    'Sultan Bazar', 'MG Bus Station'
]