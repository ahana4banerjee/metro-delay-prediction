# src/app.py
import streamlit as st
from config import DAYS_OF_WEEK
from utils import (
    load_artifacts, load_station_mapping, get_ui_station_mappings, derive_features, 
    make_prediction, get_contextual_feedback, get_route_from_station, 
    estimate_stops_between, estimate_total_delay, get_dynamic_travel_time, get_stop_sequence
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Metro Delay Predictor", page_icon="🚇", layout="wide")

# --- LOAD ARTIFACTS & MAPPINGS ---
@st.cache_resource
def init_models():
    return load_artifacts()

try:
    model, scaler, station_encoder, route_encoder = init_models()
    station_mapping = load_station_mapping()
    
    # Generate clean, unique UI names and the reverse lookup dictionary
    unique_station_names, name_to_raw_id = get_ui_station_mappings(station_encoder, station_mapping)
    available_routes = list(route_encoder.classes_)
except Exception as e:
    st.error(f"Error loading models or mapping data: {e}")
    st.stop()

# --- HEADER ---
st.title("🚇 Metro Train Delay Predictor")
st.markdown("Plan your trip and predict real-time transit delays using our **XGBoost AI Engine**.")
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📍 Trip Details")

# UI now displays ONLY unique, human-readable names
source_name = st.sidebar.selectbox("From Station (Source)", unique_station_names, index=0)

# Default destination
default_dest_idx = 5 if len(unique_station_names) > 5 else 1
dest_name = st.sidebar.selectbox("To Station (Destination)", unique_station_names, index=default_dest_idx)

if source_name == dest_name:
    st.sidebar.error("⚠️ Source and Destination cannot be the same.")
    st.error("Please select a valid destination station to proceed.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("⏱️ Commuter Parameters")

hour = st.sidebar.slider("Time of Day (Hour)", min_value=0, max_value=23, value=9)
selected_day_str = st.sidebar.selectbox("Day of the Week", list(DAYS_OF_WEEK.keys()))

st.sidebar.divider()
st.sidebar.header("⚙️ Live Network Data (Simulation)")

st.sidebar.caption("In a live app, this is fetched from metro APIs. Adjust to simulate incoming delays.")
prev_delay = st.sidebar.slider("Live Delay of Incoming Train (mins)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)

# --- AUTO-INFER & DERIVE FEATURES ---
inferred_route = get_route_from_station(source_name, available_routes)
day_val = DAYS_OF_WEEK[selected_day_str]
is_peak, is_weekend, stat_congestion = derive_features(hour, day_val, source_name)
stops_to_go = estimate_stops_between(source_name, dest_name)

# NEW: Automatically calculate the stop sequence!
stop_seq = get_stop_sequence(source_name, inferred_route)

# --- DYNAMIC TRAVEL TIME ---
avg_mins_per_stop = get_dynamic_travel_time(is_peak, is_weekend)
base_travel_time = round(stops_to_go * avg_mins_per_stop)

with st.container():
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"🛣️ **Detected Route:** {inferred_route}")
    col_b.success(f"📊 **Traffic Status:** {'Peak Hour 🔴' if is_peak else 'Off-Peak 🟢'}")
    col_c.success(f"📅 **Day Type:** {'Weekend 🌴' if is_weekend else 'Weekday 🏢'}")
st.divider()

# --- PREDICTION LOGIC ---
if st.button("Predict Trip Delay", type="primary", use_container_width=True):
    
    # 1. Lookup the raw ID from our dictionary using the clean name
    raw_station_id = name_to_raw_id[source_name]
    
    # 2. Transform that raw ID into the ML encoded format
    station_encoded = station_encoder.transform([raw_station_id])[0]
    
    route_encoded = route_encoder.transform([inferred_route])[0]
    
    input_features = {
        'station_id_encoded': station_encoded,
        'hour': hour,
        'is_peak': is_peak,
        'station_congestion': stat_congestion,
        'stop_sequence': stop_seq,
        'prev_delay': prev_delay,
        'route_encoded': route_encoded,
        'day_of_week': day_val,
        'is_weekend': is_weekend
    }
    
    # Predict Current Delay
    current_delay = make_prediction(model, scaler, input_features)
    color, status, explanation = get_contextual_feedback(current_delay, is_peak, prev_delay)
    
    # Calculate Total Trip Estimates using dynamic network logic
    total_delay = estimate_total_delay(current_delay, stops_to_go, is_peak, stat_congestion)
    total_trip_time = base_travel_time + total_delay
    
   # --- MAIN PANEL OUTPUT ---
    st.subheader(f"Trip Overview: {source_name} ➔ {dest_name}")
    st.write(f"**Distance:** {stops_to_go} stops | **Standard Travel Time:** {base_travel_time} mins")
    
    # Layout columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                <h4 style="margin:0; color: white;">Current Station Delay</h4>
                <h1 style="margin:0; font-size: 2.5rem; color: white;">{current_delay:.1f} m</h1>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div style="background-color: #2e4053; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                <h4 style="margin:0; color: white;">Total Trip Delay</h4>
                <h1 style="margin:0; font-size: 2.5rem; color: white;">{total_delay:.1f} m</h1>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div style="background-color: #1a5276; padding: 15px; border-radius: 8px; color: white; text-align: center;">
                <h4 style="margin:0; color: white;">Est. Arrival Time</h4>
                <h1 style="margin:0; font-size: 2.5rem; color: white;">{total_trip_time:.0f} m</h1>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info(f"**AI Insight:** {status} — {explanation}")
    
    min_range = max(0, round(current_delay - 2.5, 1))
    max_range = round(current_delay + 2.5, 1)
    st.caption(f"*Boarding window at {source_name}: {min_range} to {max_range} minutes delay.*")

    with st.expander("🛠️ View Input Data & ML Features "):
        st.json(input_features)