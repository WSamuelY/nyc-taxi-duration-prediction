# ============================================================
# src/04_feature_engineering.py
# Phase 4: Feature Engineering
#
# PURPOSE:
#   Transform the cleaned raw trip records into a rich feature
#   matrix that machine learning models can learn from.
#
# FEATURE GROUPS WE CREATE:
#   1. Time features    — hour, day, month, rush hour, weekend
#   2. Distance features — trip_distance, and a haversine approximation
#   3. Speed baseline   — distance ÷ avg_speed (used as both feature and baseline)
#   4. Location features — encoded PU/DO zone IDs, same-zone flag, borough IDs
#   5. Route features   — historical average duration for each PU→DO pair
#   6. Interaction features — distance × rush_hour, etc.
#
# TRAIN/TEST SPLIT LOGIC:
#   Training data = all of 2025  (fit feature encoders here ONLY)
#   Test data     = Jan–Feb 2026 (apply encoders, never fit on test)
#
# HOW TO RUN:
#   python src/04_feature_engineering.py
#
# OUTPUT:
#   data/processed/train_features.parquet
#   data/processed/test_features.parquet
#   data/processed/feature_columns.txt   (list of model input columns)
# ============================================================

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_logger, get_data_path, get_models_path,
    print_section, extract_time_features
)

logger = get_logger("04_feature_engineering")

# ── Constants ─────────────────────────────────────────────────

# NYC approximate borough center coordinates for distance proxy
# We use these to compute a rough geographic distance between
# pickup and dropoff zones when exact zone centroids aren't available.
BOROUGH_COORDS = {
    1: (40.7831, -73.9712),   # Manhattan
    2: (40.6782, -73.9442),   # Bronx (approx)
    3: (40.6501, -73.9496),   # Brooklyn
    4: (40.7282, -73.7949),   # Queens
    5: (40.5795, -74.1502),   # Staten Island
}

# Minimum number of trips a route must have to compute a
# reliable historical average (avoid noisy small samples)
MIN_ROUTE_COUNT = 50

TARGET = "trip_duration_minutes"


# ── Load Data ─────────────────────────────────────────────────

def load_data(year: int) -> pd.DataFrame:
    """Load cleaned parquet data for a given year."""
    processed_dir = get_data_path("processed")
    path = os.path.join(processed_dir, f"trips_{year}_clean.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data not found for {year}. Run 02_prepare_data.py.")

    logger.info(f"  Loading: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Feature Group 1: Time Features ───────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all temporal features from the pickup datetime.

    Rationale:
        NYC traffic is extremely time-dependent. Rush hour traffic
        can triple trip durations. These features help the model
        learn these cyclical patterns.

    Features added:
        hour, day_of_week, day_of_month, month, year,
        quarter, week_of_year, is_weekend, is_rush_hour, is_night

    Note: We use PICKUP time only (available at prediction time).
    We never use dropoff time as a feature — that would be leakage.
    """
    logger.info("  Adding time features...")
    df = extract_time_features(df, "tpep_pickup_datetime")

    # Cyclic encoding: hours are cyclical (hour 23 is close to hour 0)
    # We encode using sine and cosine to preserve this cyclical structure
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ── Feature Group 2: Distance Features ───────────────────────

def add_distance_features(df: pd.DataFrame, avg_speed_mph: float) -> pd.DataFrame:
    """
    Add distance-related features.

    Features added:
        trip_distance          — already present, kept as-is
        log_distance           — log transform reduces right skew
        baseline_duration      — distance ÷ avg_speed (our baseline model)
        distance_per_minute    — speed proxy (for context, not target leakage)

    IMPORTANT:
        'baseline_duration' is the naive prediction we must beat.
        Formula: duration_minutes = (trip_distance / avg_speed_mph) * 60

    Args:
        df:             DataFrame
        avg_speed_mph:  Average NYC taxi speed computed in EDA

    Returns:
        DataFrame with new distance features
    """
    logger.info(f"  Adding distance features (avg speed = {avg_speed_mph:.2f} mph)...")

    # Log distance: reduces the effect of a few very long trips
    # Add 1 before log to handle distance = 0 edge case
    df["log_distance"] = np.log1p(df["trip_distance"])

    # Baseline model prediction: the simplest possible estimator
    # We use this for comparison — our ML model must beat this
    df["baseline_duration"] = (df["trip_distance"] / avg_speed_mph) * 60

    return df


# ── Feature Group 3: Location Features ───────────────────────

def add_location_features(
    df: pd.DataFrame,
    zone_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add features derived from pickup and dropoff location IDs.

    Features added:
        same_zone_flag     — 1 if PU and DO are in the same zone
        pu_borough         — borough code for pickup zone (from lookup)
        do_borough         — borough code for dropoff zone
        cross_borough_flag — 1 if PU and DO boroughs differ
        pu_trip_count      — how frequently this pickup zone appears
        do_trip_count      — how frequently this dropoff zone appears

    Args:
        df:      DataFrame
        zone_df: Optional Taxi Zone Lookup DataFrame

    Returns:
        DataFrame with location features added
    """
    logger.info("  Adding location features...")

    # Same zone trips (e.g., circling a block) are usually very short
    df["same_zone_flag"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)

    # Borough mapping from zone lookup (if available)
    if zone_df is not None:
        borough_map = {
            "Manhattan": 1, "Bronx": 2, "Brooklyn": 3,
            "Queens": 4, "Staten Island": 5, "EWR": 6
        }
        zone_df = zone_df.copy()
        zone_df["borough_code"] = zone_df["Borough"].map(borough_map).fillna(0)

        pu_borough = zone_df.set_index("LocationID")["borough_code"].to_dict()
        do_borough = pu_borough.copy()

        df["pu_borough"] = df["PULocationID"].map(pu_borough).fillna(0).astype(int)
        df["do_borough"] = df["DOLocationID"].map(do_borough).fillna(0).astype(int)
        df["cross_borough_flag"] = (df["pu_borough"] != df["do_borough"]).astype(int)
    else:
        df["pu_borough"] = 0
        df["do_borough"] = 0
        df["cross_borough_flag"] = 0

    # Zone popularity: how often each zone appears as PU or DO
    # This captures zones that are structurally busy (airports, midtown)
    # IMPORTANT: compute these on TRAINING data only, then map to test
    df["pu_zone_rank"] = df["PULocationID"].map(
        df["PULocationID"].value_counts()
    ).fillna(0)
    df["do_zone_rank"] = df["DOLocationID"].map(
        df["DOLocationID"].value_counts()
    ).fillna(0)

    return df


# ── Feature Group 4: Route-Level Aggregate Features ───────────

def compute_route_features(
    train_df: pd.DataFrame
) -> dict:
    """
    Compute historical average trip durations per (PULocationID, DOLocationID) route.

    This is one of our most powerful features: if we know that trips
    from JFK to Midtown Manhattan historically take 45 minutes on average,
    that's very informative for predicting future trips on the same route.

    This is computed ONLY on training data. We then MAP these values onto
    both training and test data. This prevents data leakage.

    Args:
        train_df: 2025 training DataFrame

    Returns:
        Dictionary mapping (PU_id, DO_id) → avg_duration_minutes
    """
    logger.info("  Computing route-level historical averages (train only)...")

    route_stats = (
        train_df
        .groupby(["PULocationID", "DOLocationID"])["trip_duration_minutes"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    # Only keep routes with enough data for reliable estimates
    route_stats = route_stats[route_stats["count"] >= MIN_ROUTE_COUNT]

    logger.info(f"  Routes with ≥{MIN_ROUTE_COUNT} trips: {len(route_stats):,}")

    # Build lookup dictionaries for fast mapping
    route_mean_map   = {
        (row.PULocationID, row.DOLocationID): row["mean"]
        for row in route_stats.itertuples()
    }
    route_median_map = {
        (row.PULocationID, row.DOLocationID): row["median"]
        for row in route_stats.itertuples()
    }

    return route_mean_map, route_median_map


def add_route_features(
    df: pd.DataFrame,
    route_mean_map: dict,
    route_median_map: dict,
    global_mean: float
) -> pd.DataFrame:
    """
    Map precomputed route statistics onto a DataFrame.

    For routes not seen in training, we fall back to the global mean
    (this handles the cold-start problem for rare routes).

    Args:
        df:               DataFrame to add features to
        route_mean_map:   Dict of (PU, DO) → mean duration
        route_median_map: Dict of (PU, DO) → median duration
        global_mean:      Fallback value for unseen routes

    Returns:
        DataFrame with 'route_mean_duration' and 'route_median_duration' columns
    """
    logger.info("  Mapping route features...")

    keys = list(zip(df["PULocationID"], df["DOLocationID"]))

    df["route_mean_duration"] = [
        route_mean_map.get(k, global_mean) for k in keys
    ]
    df["route_median_duration"] = [
        route_median_map.get(k, global_mean) for k in keys
    ]

    coverage = sum(1 for k in keys if k in route_mean_map) / len(keys) * 100
    logger.info(f"  Route feature coverage: {coverage:.1f}% of trips matched")

    return df


# ── Feature Group 5: Interaction Features ────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between important feature pairs.

    Interaction features capture combined effects that individual
    features miss. For example:
    - A long trip DURING rush hour is much worse than either alone
    - Airport trips have a very different distance-duration relationship

    Features added:
        distance_x_rush    — distance × is_rush_hour
        distance_x_weekend — distance × is_weekend
        distance_x_hour    — distance × hour (captures speed by time)
    """
    logger.info("  Adding interaction features...")

    df["distance_x_rush"]    = df["trip_distance"] * df["is_rush_hour"]
    df["distance_x_weekend"] = df["trip_distance"] * df["is_weekend"]
    df["distance_x_hour"]    = df["trip_distance"] * df["hour"]

    return df


# ── Define Final Feature Set ──────────────────────────────────

def get_feature_columns() -> list:
    """
    Return the definitive list of columns to use as model inputs (X).

    This is centralized here so all scripts agree on the same features.
    Any column not in this list is excluded from modeling.

    Returns:
        List of feature column names
    """
    return [
        # Raw numeric features
        "trip_distance",
        "log_distance",
        "passenger_count",
        "fare_amount",          # Proxy for meter-recorded distance/time
        "RatecodeID",
        "payment_type",

        # Location features
        "PULocationID",
        "DOLocationID",
        "same_zone_flag",
        "pu_borough",
        "do_borough",
        "cross_borough_flag",
        "pu_zone_rank",
        "do_zone_rank",

        # Time features (raw)
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_rush_hour",
        "is_night",

        # Time features (cyclic encoded)
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",

        # Baseline and route features
        "baseline_duration",
        "route_mean_duration",
        "route_median_duration",

        # Interaction features
        "distance_x_rush",
        "distance_x_weekend",
        "distance_x_hour",
    ]


# ── Main Pipeline ─────────────────────────────────────────────

def run_feature_engineering():
    """
    Full feature engineering pipeline:
    1. Load train (2025) and test (2026) data
    2. Add time, distance, location, route, and interaction features
    3. Compute route lookups on train only
    4. Save final feature matrices
    """
    print_section("Phase 4: Feature Engineering")

    # ── Load Data ─────────────────────────────────────────────
    train_df = load_data(2025)
    test_df  = load_data(2026)

    # ── Load supporting files ─────────────────────────────────
    # Zone lookup (optional but recommended)
    zone_path = os.path.join(get_data_path("raw"), "taxi_zone_lookup.csv")
    zone_df = pd.read_csv(zone_path) if os.path.exists(zone_path) else None
    if zone_df is not None:
        logger.info(f"  Zone lookup loaded: {len(zone_df)} zones")

    # Average speed from EDA
    speed_path = os.path.join(get_data_path("processed"), "avg_speed.txt")
    if os.path.exists(speed_path):
        with open(speed_path) as f:
            avg_speed = float(f.read().strip())
    else:
        avg_speed = 12.5   # Reasonable NYC taxi default if EDA not run
    logger.info(f"  Using average speed: {avg_speed:.2f} mph")

    # ── Feature Engineering — TRAINING DATA ───────────────────
    print_section("Engineering Training Features (2025)")

    train_df = add_time_features(train_df)
    train_df = add_distance_features(train_df, avg_speed)
    train_df = add_location_features(train_df, zone_df)

    # Route features: computed on training data ONLY
    route_mean_map, route_median_map = compute_route_features(train_df)
    global_mean = train_df[TARGET].mean()

    train_df = add_route_features(train_df, route_mean_map, route_median_map, global_mean)
    train_df = add_interaction_features(train_df)

    # ── Feature Engineering — TEST DATA ───────────────────────
    print_section("Engineering Test Features (Jan–Feb 2026)")

    test_df = add_time_features(test_df)
    test_df = add_distance_features(test_df, avg_speed)

    # For test set location features, use the zone lookup
    # but DO NOT recompute zone_rank from test data (leakage!)
    test_df = add_location_features(test_df, zone_df)

    # Replace zone rank with training-data-based ranks
    train_pu_ranks = train_df["PULocationID"].value_counts().to_dict()
    train_do_ranks = train_df["DOLocationID"].value_counts().to_dict()
    test_df["pu_zone_rank"] = test_df["PULocationID"].map(train_pu_ranks).fillna(0)
    test_df["do_zone_rank"] = test_df["DOLocationID"].map(train_do_ranks).fillna(0)

    # Apply training-data route maps to test set
    test_df = add_route_features(test_df, route_mean_map, route_median_map, global_mean)
    test_df = add_interaction_features(test_df)

    # ── Save Feature Matrices ──────────────────────────────────
    FEATURE_COLS = get_feature_columns()
    processed_dir = get_data_path("processed")

    logger.info(f"\n  Final feature count: {len(FEATURE_COLS)}")

    # Save training features + target
    train_out = train_df[FEATURE_COLS + [TARGET]].copy()
    train_path = os.path.join(processed_dir, "train_features.parquet")
    train_out.to_parquet(train_path, index=False)
    logger.info(f"  ✓ Training features saved: {train_path}")
    logger.info(f"    Shape: {train_out.shape}")

    # Save test features + target
    test_out = test_df[FEATURE_COLS + [TARGET]].copy()
    test_path = os.path.join(processed_dir, "test_features.parquet")
    test_out.to_parquet(test_path, index=False)
    logger.info(f"  ✓ Test features saved: {test_path}")
    logger.info(f"    Shape: {test_out.shape}")

    # Save the feature column list — used by all modeling scripts
    feat_list_path = os.path.join(processed_dir, "feature_columns.txt")
    with open(feat_list_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))
    logger.info(f"  ✓ Feature list saved: {feat_list_path}")

    # Save route maps for use in API / Streamlit app
    models_dir = get_models_path()
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(
        {
            "route_mean_map": route_mean_map,
            "route_median_map": route_median_map,
            "global_mean": global_mean,
            "avg_speed_mph": avg_speed,
        },
        os.path.join(models_dir, "route_maps.joblib")
    )
    logger.info(f"  ✓ Route maps saved to models/route_maps.joblib")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 4: FEATURE ENGINEERING")
    print("="*60)

    run_feature_engineering()

    print("\n✅ Phase 4 complete. Check data/processed/ for feature files.")
    print("   Next step: python src/05_modeling.py\n")
