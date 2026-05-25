# ============================================================
# src/streamlit_app.py  —  Interactive Streamlit Dashboard
# Run: streamlit run src/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_models_path, get_data_path, get_figures_path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Duration Predictor",
    page_icon="🚕",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir = get_models_path()
    lgbm  = joblib.load(os.path.join(models_dir, "lightgbm.joblib"))
    maps  = joblib.load(os.path.join(models_dir, "route_maps.joblib"))
    feat  = os.path.join(get_data_path("processed"), "feature_columns.txt")
    with open(feat) as f:
        cols = [l.strip() for l in f.readlines()]
    return lgbm, maps, cols

try:
    lgbm_model, route_maps, FEATURE_COLS = load_artifacts()
    READY = True
except:
    READY = False

# ── NYC Zone names (top 30 for dropdown) ─────────────────────
ZONES = {
    1:"Newark Airport", 132:"JFK Airport", 138:"LaGuardia Airport",
    161:"Midtown Center", 162:"Midtown East", 163:"Midtown North",
    230:"Times Sq/Theatre District", 236:"Upper East Side N",
    237:"Upper East Side S", 239:"Upper West Side N",
    170:"Murray Hill", 48:"Clinton East", 186:"Penn Station/Madison Sq W",
    141:"Lenox Hill East", 142:"Lenox Hill West", 234:"Union Sq",
    113:"Greenwich Village North", 114:"Greenwich Village South",
    79:"East Village", 43:"Central Park",
    261:"Woodside", 76:"East Elmhurst", 97:"Flatbush/Ditmas Park",
    112:"Greenwich Village North", 144:"Lincoln Square East"
}

# ── Helper: build feature row ─────────────────────────────────
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
        "route_median_duration":route_maps["route_median_map"].get(route_key, global_mean),
        "hour":dt.hour,"day_of_week":dt.weekday(),"day_of_month":dt.day,
        "month":dt.month,"year":dt.year,"quarter":(dt.month-1)//3+1,
        "week_of_year":dt.isocalendar()[1],
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
    return pd.DataFrame([row])[FEATURE_COLS].astype("float32")

# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════

st.title("🚕 NYC Yellow Taxi — Trip Duration Predictor")
st.markdown("*Predict how long your trip will take before you get in the cab.*")

if not READY:
    st.error("⚠️ Models not loaded. Run `python src/05_modeling.py` first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ── TAB 1: Prediction ─────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Trip Details")
        zone_options = {v: k for k, v in ZONES.items()}

        pu_name = st.selectbox("📍 Pickup Zone", list(ZONES.values()), index=1)
        do_name = st.selectbox("🏁 Dropoff Zone", list(ZONES.values()), index=5)
        pu_id   = zone_options[pu_name]
        do_id   = zone_options[do_name]

        distance    = st.slider("📏 Trip Distance (miles)", 0.5, 30.0, 3.0, 0.5)
        passengers  = st.selectbox("👥 Passengers", [1,2,3,4,5,6], index=0)
        rate_code   = st.selectbox("💳 Rate Code", [1,2,3,4,5,6],
                                   format_func=lambda x: {1:"Standard",2:"JFK",3:"Newark",
                                                           4:"Nassau/Westchester",5:"Negotiated",6:"Group"}[x])

    with col2:
        st.subheader("Time of Pickup")
        pickup_date = st.date_input("📅 Date", value=datetime(2026, 2, 15))
        pickup_time = st.time_input("🕐 Time", value=datetime(2026, 2, 15, 8, 30).time())
        dt          = datetime.combine(pickup_date, pickup_time)
        fare_est    = st.number_input("💵 Estimated Fare ($)", min_value=0.0,
                                      value=round(distance * 3.5, 2), step=0.5)

        st.markdown("---")
        rush = (7<=dt.hour<=9) or (16<=dt.hour<=19)
        weekend = dt.weekday() >= 5
        st.info(
            f"**Context:**  \n"
            f"{'🔴 Rush Hour' if rush else '🟢 Off-Peak'}  |  "
            f"{'🏖️ Weekend' if weekend else '💼 Weekday'}  |  "
            f"Hour {dt.hour:02d}:00"
        )

    st.markdown("---")
    if st.button("🚀 Predict Trip Duration", use_container_width=True, type="primary"):
        X = build_features(pu_id, do_id, distance, dt, passengers, rate_code, fare_est)
        pred     = float(np.clip(lgbm_model.predict(X)[0], 1, 300))
        baseline = float((distance / route_maps["avg_speed_mph"]) * 60)

        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 ML Prediction", f"{pred:.1f} min",
                  delta=f"{pred-baseline:+.1f} vs baseline")
        c2.metric("📏 Baseline Estimate", f"{baseline:.1f} min")
        c3.metric("🏁 ETA",
                  f"{(datetime.combine(pickup_date, pickup_time) + pd.Timedelta(minutes=pred)).strftime('%H:%M')}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text": "Predicted Duration (minutes)"},
            gauge={
                "axis": {"range": [0, 120]},
                "bar": {"color": "#1E88E5"},
                "steps": [
                    {"range": [0, 20],  "color": "#C8E6C9"},
                    {"range": [20, 45], "color": "#FFF9C4"},
                    {"range": [45, 120],"color": "#FFCDD2"},
                ],
                "threshold": {"line": {"color":"red","width":3}, "value": baseline}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Model Performance ──────────────────────────────────
with tab2:
    st.subheader("Model Comparison — Test Set (Jan–Feb 2026)")

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "model_report.md"
    )
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.warning("Run `python src/06_evaluation.py` to generate the model report.")

    # Show saved figures
    figures_dir = get_figures_path()
    for fname, title in [
        ("07_model_comparison.png", "Model Comparison"),
        ("08_feature_importance.png", "Feature Importance"),
        ("09_actual_vs_predicted.png", "Actual vs Predicted"),
        ("10_error_analysis.png", "Error Analysis"),
    ]:
        fpath = os.path.join(figures_dir, fname)
        if os.path.exists(fpath):
            st.subheader(title)
            st.image(fpath, use_column_width=True)

# ── TAB 3: About ──────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## About This Project

    **Problem:** Predict NYC yellow taxi trip duration in minutes using only
    information available at the time of pickup.

    **Data:** NYC TLC Yellow Taxi Trip Records — Jan 2025 to Feb 2026 (14 months)

    **Models Trained:**
    | Model | Type |
    |-------|------|
    | Linear Regression | Baseline linear |
    | Ridge Regression  | Regularized linear |
    | LightGBM          | Gradient boosting (best) |
    | MLP Neural Network| Deep learning |

    **Success Criteria:**
    - RMSE < 5 minutes ✅
    - MAE < 3 minutes ✅
    - R² > 0.80 ✅
    - Beat distance ÷ speed baseline ✅

    **Tech Stack:** Python · Pandas · LightGBM · Scikit-learn · FastAPI · Streamlit
    """)
