# src/app.py
import streamlit as st
import pandas as pd
from config import DAYS_OF_WEEK, WEATHER_CONDITIONS, EVENT_INTENSITY
from utils import (
    load_artifacts, load_station_mapping, get_ui_station_mappings, derive_features, 
    make_prediction, get_contextual_feedback, estimate_total_delay
)

# --- PAGE CONFIG ---\
st.set_page_config(page_title="Hybrid Metro Predictor", page_icon="🚇", layout="wide")

# --- LOAD ARTIFACTS & MAPPINGS ---\
@st.cache_resource
def init_models():
    return load_artifacts()

try:
    model, scaler, station_encoder, route_encoder = init_models()
    station_mapping = load_station_mapping()
    unique_station_names, name_to_raw_id = get_ui_station_mappings(station_encoder, station_mapping)
except Exception as e:
    st.error(f"Error loading framework architectures: {e}")
    st.stop()

# --- HEADER ---\
st.title("🚇 Hybrid Intelligent Metro Delay Framework")
st.markdown("""
*Research Prototype:* This dashboard integrates an **XGBoost Machine Learning Schedule Predictor** with a 
**Domain-Driven Operational Adjustment Score (OAS)** to predict delays under severe environmental disruptions.
""")
st.divider()

# --- SIDEBAR CONTROLS ---\
st.sidebar.header("📍 1. Schedule & Routing")
selected_day = st.sidebar.selectbox("Day of Week", list(DAYS_OF_WEEK.keys()))
hour_of_day = st.sidebar.slider("Hour of Day (24H)", min_value=6, max_value=23, value=8)

source_station = st.sidebar.selectbox("Origin Station", unique_station_names, index=0)
dest_station = st.sidebar.selectbox("Destination Station", unique_station_names, index=len(unique_station_names)-1)

st.sidebar.divider()
st.sidebar.header("🌪️ 2. Environmental Disruptions")
st.sidebar.markdown("*Test the Hybrid framework's resilience.*")
weather_ui = st.sidebar.selectbox("Current Weather", list(WEATHER_CONDITIONS.keys()))
event_ui = st.sidebar.selectbox("Event & Crowding", list(EVENT_INTENSITY.keys()))

st.sidebar.divider()
prev_delay_input = st.sidebar.number_input("Incoming Network Delay (Mins)", min_value=0.0, max_value=30.0, value=0.0, step=0.5,
                                           help="Simulates if the train is already late arriving at your origin.")

# --- MAIN PREDICTION LOGIC ---\
if st.button("🚀 Run Hybrid Delay Prediction", use_container_width=True):
    if source_station == dest_station:
        st.warning("Origin and Destination cannot be the same.")
    else:
        with st.spinner("Executing Decoupled Feature Architecture..."):
            
            # Map Inputs
            day_int = DAYS_OF_WEEK[selected_day]
            weather_val = WEATHER_CONDITIONS[weather_ui]
            event_val = EVENT_INTENSITY[event_ui]
            source_raw_id = name_to_raw_id.get(source_station, 'Unknown')
            
            # 1. Predict Current Station Delay (Breakdown Analysis)
            features_df = derive_features(
                source_raw_id, hour_of_day, day_int, prev_delay_input, 
                station_encoder, route_encoder, source_station,
                weather_val, event_val
            )
            
            current_delay, xgb_ml_pred, oas_physics_score = make_prediction(features_df, model, scaler)
            
           # 2. Predict Total Trip Propagation
            total_delay, trip_time, stops = estimate_total_delay(
                source_station, dest_station, hour_of_day, day_int, 
                station_encoder, route_encoder, name_to_raw_id,
                weather_val, event_val, model, scaler, prev_delay_input # <-- Added prev_delay_input
            )

        # --- RESULTS DASHBOARD ---\
        st.success(f"Trip Details: **{source_station}** ➔ **{dest_station}** ({stops} stops)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style="background-color: #1E3A8A; padding: 15px; border-radius: 8px; color: white; text-align: center; border: 2px solid #3B82F6;">
                    <h4 style="margin:0; color: #93C5FD;">Origin Delay</h4>
                    <h1 style="margin:0; font-size: 2.8rem; color: white;">{current_delay:.1f} <span style="font-size:1.2rem;">min</span></h1>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div style="background-color: #374151; padding: 15px; border-radius: 8px; color: white; text-align: center; border: 2px solid #6B7280;">
                    <h4 style="margin:0; color: #D1D5DB;">Total Trip Delay</h4>
                    <h1 style="margin:0; font-size: 2.8rem; color: white;">{total_delay:.1f} <span style="font-size:1.2rem;">min</span></h1>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div style="background-color: #064E3B; padding: 15px; border-radius: 8px; color: white; text-align: center; border: 2px solid #10B981;">
                    <h4 style="margin:0; color: #6EE7B7;">Total Travel Time</h4>
                    <h1 style="margin:0; font-size: 2.8rem; color: white;">{(trip_time + total_delay):.0f} <span style="font-size:1.2rem;">min</span></h1>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.info(get_contextual_feedback(current_delay, oas_physics_score, features_df.iloc[0]['is_peak'], weather_val))

        # --- RESEARCH METRICS BREAKDOWN (The "Novelty" display) ---
        st.markdown("### 🔬 Hybrid Model Decoupling Breakdown")
        st.markdown("*This section demonstrates the internal blending of the Machine Learning baseline with the Operational Adjustment Score.*")
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("🤖 XGBoost (Schedule Data)", f"{xgb_ml_pred:.2f} min", help="Prediction based purely on time and routing.")
        b_col2.metric("🌪️ OAS (Physics/Environment)", f"+ {oas_physics_score:.2f} impact", help="Heuristic calculation of weather and events.")
        
        recovery_pct = features_df.iloc[0]['recovery_factor'] * 100
        b_col3.metric("⏱️ System Resilience (Recovery)", f"- {recovery_pct:.0f}%", help="Trains catch up time during off-peak or terminal approaches.")