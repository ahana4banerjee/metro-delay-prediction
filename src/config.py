# src/config.py

FEATURE_COLUMNS = [
    'station_id_encoded', 'hour', 'is_peak', 'station_congestion', 
    'stop_sequence', 'prev_delay', 'route_encoded', 
    'day_of_week', 'is_weekend'
]

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

# --- NEW: REAL METRO LINE SEQUENCES FOR DISTANCE CALCULATION ---

RED_LINE = [
    'Miyapur', 'JNTU College', 'KPHB Colony', 'Kukatpally', 'Balanagar', 
    'Moosapet', 'Bharat Nagar', 'Erragadda', 'ESI Hospital', 'SR Nagar', 
    'Ameerpet', 'Punjagutta', 'Irrum Manzil', 'Khairatabad', 'Lakdikapul', 
    'Assembly', 'Nampally', 'Gandhi Bhavan', 'Osmania Medical College', 
    'MG Bus Station', 'Malakpet', 'New Market', 'Musarambagh', 'Dilsukhnagar', 
    'Chaitanyapuri', 'Victoria Memorial', 'LB Nagar'
]

BLUE_LINE = [
    'Nagole', 'Uppal', 'Stadium', 'NGRI', 'Habsiguda', 'Tarnaka', 'Mettuguda', 
    'Secunderabad East', 'Parade Ground', 'Paradise', 'Rasoolpura', 'Prakash Nagar', 
    'Begumpet', 'Ameerpet', 'Madhura Nagar', 'Yusufguda', 'Road No 5 Jubilee Hills', 
    'Jubilee Hills Check Post', 'Peddamma Temple', 'Madhapur', 'Durgam Cheruvu', 
    'HITEC City', 'Raidurg'
]

GREEN_LINE = [
    'JBS Parade Ground', 'Secunderabad West', 'Gandhi Hospital', 'Musheerabad', 
    'RTC Cross Roads', 'Chikkadpally', 'Narayanaguda', 'Sultan Bazar', 'MG Bus Station'
]