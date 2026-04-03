# src/app.py
import streamlit as st
import pandas as pd
from config import DAYS_OF_WEEK
from utils import load_artifacts, derive_features, make_prediction, get_contextual_feedback

# --- PAGE CONFIG ---
st.set_page_config(page_title="Metro Delay Predictor", page_icon="🚇", layout="wide")

# --- LOAD ARTIFACTS ---
@st.cache_resource # Caches models so they don't reload on every button click
def init_models():
    return load_artifacts()

try:
    model, scaler, station_encoder, route_encoder = init_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- HEADER ---
st.title("🚇 Metro Train Delay Predictor")
st.markdown("Predict real-time transit delays using the **XGBoost** machine learning model.")
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("⏱️ Trip Parameters")

# Extract available stations and routes from the encoders to populate dropdowns
available_stations = sorted(station_encoder.classes_)
available_routes = sorted(route_encoder.classes_)

# User Inputs
selected_station = st.sidebar.selectbox("Current Station", available_stations)
selected_route = st.sidebar.selectbox("Route / Line", available_routes)
hour = st.sidebar.slider("Time of Day (Hour)", min_value=0, max_value=23, value=9)
stop_seq = st.sidebar.number_input("Stop Sequence Number", min_value=1, max_value=50, value=5)
prev_delay = st.sidebar.slider("Delay at Previous Station (mins)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
selected_day_str = st.sidebar.selectbox("Day of the Week", list(DAYS_OF_WEEK.keys()))

# --- PREDICTION LOGIC ---
if st.button("Predict Delay", type="primary"):
    
    # 1. Transform raw inputs using encoders and config
    station_encoded = station_encoder.transform([selected_station])[0]
    route_encoded = route_encoder.transform([selected_route])[0]
    day_val = DAYS_OF_WEEK[selected_day_str]
    
    # 2. Auto-compute derived features
    is_peak, is_weekend, stat_congestion = derive_features(hour, day_val, selected_station)
    
    # 3. Compile the feature dictionary
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
    
    # 4. Predict
    prediction = make_prediction(model, scaler, input_features)
    color, status, explanation = get_contextual_feedback(prediction, is_peak, prev_delay)
    
    # --- MAIN PANEL OUTPUT ---
    st.subheader("Prediction Results")
    
    # Layout columns for neat output
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Colored metric display using HTML/Markdown
        st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h1 style="margin:0; font-size: 3rem; color: white;">{prediction}</h1>
                <p style="margin:0; font-size: 1.2rem;">Minutes Delay</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write(f"**Status:** {status}")
        st.info(f"**AI Insight:** {explanation}")
        
        # Estimate a confidence range based on standard XGBoost MAE (~2.5 mins)
        min_range = max(0, round(prediction - 2.5, 1))
        max_range = round(prediction + 2.5, 1)
        st.caption(f"*Expected arrival window: {min_range} to {max_range} minutes delay.*")

    # Extra features: Show the data used (Great for presentations)
    with st.expander("View Input Data & Derived Features"):
        st.json(input_features)