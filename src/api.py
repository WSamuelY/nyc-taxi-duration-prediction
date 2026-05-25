# ============================================================
# src/api.py  —  FastAPI Prediction Endpoint
# Run: uvicorn src.api:app --reload
# ============================================================

import os, sys, numpy as np, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import extract_time_features, get_models_path, get_data_path

app = FastAPI(
    title="NYC Taxi Trip Duration API",
    description="Predicts yellow taxi trip duration in minutes",
    version="1.0.0"
)

# ── Load models & artifacts at startup ───────────────────────
models_dir = get_models_path()

try:
    lgbm_model  = joblib.load(os.path.join(models_dir, "lightgbm.joblib"))
    route_maps  = joblib.load(os.path.join(models_dir, "route_maps.joblib"))
    feat_path   = os.path.join(get_data_path("processed"), "feature_columns.txt")
    with open(feat_path) as f:
        FEATURE_COLS = [l.strip() for l in f.readlines()]
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    LOAD_ERROR = str(e)


# ── Request / Response Schemas ────────────────────────────────

class TripRequest(BaseModel):
    pickup_location_id:  int   = Field(..., ge=1, le=265, example=132)
    dropoff_location_id: int   = Field(..., ge=1, le=265, example=236)
    trip_distance:       float = Field(..., gt=0, example=2.5)
    pickup_datetime:     str   = Field(..., example="2025-06-15 08:30:00")
    passenger_count:     int   = Field(1, ge=1, le=8)
    rate_code_id:        int   = Field(1, ge=1, le=6)
    payment_type:        int   = Field(1, ge=1, le=6)
    fare_amount:         float = Field(10.0, ge=0)

class TripResponse(BaseModel):
    predicted_duration_minutes: float
    baseline_duration_minutes:  float
    model_used: str
    pickup_datetime: str


# ── Helper: Build feature row ─────────────────────────────────

def build_feature_row(req: TripRequest) -> pd.DataFrame:
    avg_speed = route_maps["avg_speed_mph"]
    route_key = (req.pickup_location_id, req.dropoff_location_id)
    global_mean = route_maps["global_mean"]

    row = {
        "trip_distance":        req.trip_distance,
        "log_distance":         float(np.log1p(req.trip_distance)),
        "passenger_count":      float(req.passenger_count),
        "fare_amount":          req.fare_amount,
        "RatecodeID":           float(req.rate_code_id),
        "payment_type":         float(req.payment_type),
        "PULocationID":         req.pickup_location_id,
        "DOLocationID":         req.dropoff_location_id,
        "same_zone_flag":       int(req.pickup_location_id == req.dropoff_location_id),
        "pu_borough":           0, "do_borough": 0, "cross_borough_flag": 0,
        "pu_zone_rank":         0, "do_zone_rank": 0,
        "baseline_duration":    (req.trip_distance / avg_speed) * 60,
        "route_mean_duration":  route_maps["route_mean_map"].get(route_key, global_mean),
        "route_median_duration":route_maps["route_median_map"].get(route_key, global_mean),
    }

    # Time features
    dt = pd.to_datetime(req.pickup_datetime)
    row.update({
        "hour": dt.hour, "day_of_week": dt.dayofweek,
        "day_of_month": dt.day, "month": dt.month, "year": dt.year,
        "quarter": dt.quarter, "week_of_year": dt.isocalendar()[1],
        "is_weekend":   int(dt.dayofweek >= 5),
        "is_rush_hour": int((7 <= dt.hour <= 9) or (16 <= dt.hour <= 19)),
        "is_night":     int(dt.hour >= 22 or dt.hour <= 5),
        "hour_sin":  float(np.sin(2*np.pi*dt.hour/24)),
        "hour_cos":  float(np.cos(2*np.pi*dt.hour/24)),
        "dow_sin":   float(np.sin(2*np.pi*dt.dayofweek/7)),
        "dow_cos":   float(np.cos(2*np.pi*dt.dayofweek/7)),
        "month_sin": float(np.sin(2*np.pi*dt.month/12)),
        "month_cos": float(np.cos(2*np.pi*dt.month/12)),
        "distance_x_rush":    req.trip_distance * int((7<=dt.hour<=9)or(16<=dt.hour<=19)),
        "distance_x_weekend": req.trip_distance * int(dt.dayofweek >= 5),
        "distance_x_hour":    req.trip_distance * dt.hour,
    })

    df = pd.DataFrame([row])
    return df[FEATURE_COLS].astype("float32")


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health():
    if not MODELS_LOADED:
        raise HTTPException(503, f"Models not loaded: {LOAD_ERROR}")
    return {"status": "ok", "model": "LightGBM", "features": len(FEATURE_COLS)}


@app.post("/predict", response_model=TripResponse)
def predict(req: TripRequest):
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded. Run 05_modeling.py first.")
    try:
        X = build_feature_row(req)
        duration = float(np.clip(lgbm_model.predict(X)[0], 1, 300))
        baseline = float((req.trip_distance / route_maps["avg_speed_mph"]) * 60)
        return TripResponse(
            predicted_duration_minutes=round(duration, 2),
            baseline_duration_minutes=round(baseline, 2),
            model_used="LightGBM",
            pickup_datetime=req.pickup_datetime
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")


@app.get("/")
def root():
    return {"message": "NYC Taxi Duration API. POST to /predict or visit /docs"}
