
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚕",
    layout="wide"
)

# Get project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

@st.cache_resource
def load_artifacts():
    lgbm  = joblib.load(os.path.join(MODELS_DIR, "lightgbm.joblib"))
    maps  = joblib.load(os.path.join(MODELS_DIR, "route_maps.joblib"))
    feat  = os.path.join(MODELS_DIR, "feature_columns.txt")
    with open(feat) as f:
        cols = [l.strip() for l in f.readlines()]
    return lgbm, maps, cols

try:
    lgbm_model, route_maps, FEATURE_COLS = load_artifacts()
    READY = True
except Exception as e:
    READY = False
    ERROR = str(e)

ZONES = {
    132:"JFK Airport", 138:"LaGuardia Airport",
    161:"Midtown Center", 162:"Midtown East",
    230:"Times Sq/Theatre District",
    236:"Upper East Side N", 237:"Upper East Side S",
    170:"Murray Hill", 186:"Penn Station",
    79:"East Village", 43:"Central Park",
    1:"Newark Airport", 261:"Woodside",
    48:"Clinton East", 113:"Greenwich Village N",
}

def build_features(pu_id, do_id, distance, dt, passengers, rate_code, fare):
    avg_speed   = route_maps["avg_speed_mph"]
    global_mean = route_maps["global_mean"]
    route_key   = (pu_id, do_id)
    row = {
        "trip_distance":        distance,
        "log_distance":         np.log1p(distance),
        "passenger_count":      float(passengers),
        "fare_amount":          fare,
        "RatecodeID":           float(rate_code),
        "payment_type":         1.0,
        "PULocationID":         pu_id,
        "DOLocationID":         do_id,
        "same_zone_flag":       int(pu_id == do_id),
        "pu_borough":0,"do_borough":0,"cross_borough_flag":0,
        "pu_zone_rank":0,"do_zone_rank":0,
        "baseline_duration":    (distance / avg_speed) * 60,
        "route_mean_duration":  route_maps["route_mean_map"].get(route_key, global_mean),
        "route_median_duration":route_maps["route_mean_map"].get(route_key, global_mean),
        "hour":dt.hour,"day_of_week":dt.weekday(),
        "day_of_month":dt.day,"month":dt.month,
        "year":dt.year,"quarter":(dt.month-1)//3+1,
        "week_of_year":int(dt.isocalendar()[1]),
        "is_weekend":   int(dt.weekday()>=5),
        "is_rush_hour": int((7<=dt.hour<=9)or(16<=dt.hour<=19)),
        "is_night":     int(dt.hour>=22 or dt.hour<=5),
        "hour_sin":  np.sin(2*np.pi*dt.hour/24),
        "hour_cos":  np.cos(2*np.pi*dt.hour/24),
        "dow_sin":   np.sin(2*np.pi*dt.weekday()/7),
        "dow_cos":   np.cos(2*np.pi*dt.weekday()/7),
        "month_sin": np.sin(2*np.pi*dt.month/12),
        "month_cos": np.cos(2*np.pi*dt.month/12),
        "distance_x_rush":    distance*int((7<=dt.hour<=9)or(16<=dt.hour<=19)),
        "distance_x_weekend": distance*int(dt.weekday()>=5),
        "distance_x_hour":    distance*dt.hour,
    }
    df = pd.DataFrame([row])
    df = df[FEATURE_COLS]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.astype("float32")

# ── UI ────────────────────────────────────────────────────────
st.title("🚕 NYC Yellow Taxi — Trip Duration Predictor")
st.markdown("*Predict how long your trip will take before you get in the cab.*")

if not READY:
    st.error(f"Models not loaded: {ERROR}")
    st.stop()

tab1, tab2 = st.tabs(["🔮 Predict", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trip Details")
        zone_options = {v: k for k, v in ZONES.items()}
        pu_name = st.selectbox("📍 Pickup Zone",  list(ZONES.values()), index=0)
        do_name = st.selectbox("🏁 Dropoff Zone", list(ZONES.values()), index=2)
        pu_id   = zone_options[pu_name]
        do_id   = zone_options[do_name]
        distance   = st.slider("📏 Trip Distance (miles)", 0.5, 30.0, 3.0, 0.5)
        passengers = st.selectbox("👥 Passengers", [1,2,3,4,5,6])
        rate_code  = st.selectbox("💳 Rate Code", [1,2,3],
                       format_func=lambda x: {1:"Standard",2:"JFK",3:"Newark"}[x])

    with col2:
        st.subheader("Time of Pickup")
        pickup_date = st.date_input("📅 Date", value=datetime(2026, 2, 15))
        pickup_time = st.time_input("🕐 Time", value=datetime(2026,2,15,8,30).time())
        dt          = datetime.combine(pickup_date, pickup_time)
        fare_est    = st.number_input("💵 Estimated Fare ($)",
                        min_value=0.0, value=round(distance*3.5,2), step=0.5)
        rush    = (7<=dt.hour<=9) or (16<=dt.hour<=19)
        weekend = dt.weekday() >= 5
        st.info(
            f"**Context:**  \n"
            f"{'🔴 Rush Hour' if rush else '🟢 Off-Peak'}  |  "
            f"{'🏖️ Weekend' if weekend else '💼 Weekday'}  |  "
            f"Hour {dt.hour:02d}:00"
        )

    st.markdown("---")
    if st.button("🚀 Predict Trip Duration", use_container_width=True, type="primary"):
        X    = build_features(pu_id, do_id, distance, dt, passengers, rate_code, fare_est)
        pred = float(np.clip(lgbm_model.predict(X)[0], 1, 300))
        base = float((distance / route_maps["avg_speed_mph"]) * 60)
        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 ML Prediction", f"{pred:.1f} min", delta=f"{pred-base:+.1f} vs baseline")
        c2.metric("📏 Baseline", f"{base:.1f} min")
        arrival = datetime.combine(pickup_date, pickup_time)
        import pandas as pd_
        eta = arrival + pd_.Timedelta(minutes=pred)
        c3.metric("🏁 ETA", eta.strftime("%H:%M"))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Duration (minutes)"},
            gauge={
                "axis": {"range": [0, 120]},
                "bar":  {"color": "#1E88E5"},
                "steps": [
                    {"range": [0,20],  "color": "#C8E6C9"},
                    {"range": [20,45], "color": "#FFF9C4"},
                    {"range": [45,120],"color": "#FFCDD2"},
                ],
                "threshold": {"line":{"color":"red","width":3}, "value": base}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    ## About This Project
    - **Data:** NYC TLC Yellow Taxi — Oct 2025 to Feb 2026
    - **Training rows:** 11,485,385
    - **Test rows:** 6,633,847
    - **Best Model:** LightGBM

    | Metric | Target | Achieved |
    |--------|--------|----------|
    | RMSE | < 5 min | 4.74 min ✅ |
    | MAE  | < 3 min | 2.29 min ✅ |
    | R²   | > 0.80  | 0.88 ✅ |
    | Within ±5 min | - | 87.7% 🌟 |
    """)
