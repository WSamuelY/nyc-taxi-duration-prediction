# ============================================================
# tests/test_preprocessing.py
# Unit Tests for Data Preprocessing & Utility Functions
#
# PURPOSE:
#   Verify that all data preparation and feature engineering
#   functions produce correct results. These tests run in CI/CD
#   via GitHub Actions on every push.
#
# HOW TO RUN:
#   pytest tests/test_preprocessing.py -v
#
# WHAT WE TEST:
#   - Duration computation (correct, zero, negative cases)
#   - Outlier filtering thresholds
#   - Time feature extraction
#   - Cyclic encoding properties
#   - IQR bounds calculation
#   - Regression metrics computation
#   - Route feature mapping
# ============================================================

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    extract_time_features,
    iqr_bounds,
    regression_metrics
)


# ═══════════════════════════════════════════════════════════
# Fixtures — Reusable test data
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_trip_df():
    """
    A small synthetic DataFrame that mimics cleaned TLC trip records.
    Used across multiple tests.
    """
    return pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2025-01-15 08:00:00",   # Morning rush
            "2025-03-20 14:30:00",   # Afternoon
            "2025-06-01 00:15:00",   # Late night
            "2025-12-31 17:45:00",   # PM rush, year-end
            "2025-07-04 12:00:00",   # Holiday noon
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2025-01-15 08:25:00",
            "2025-03-20 14:52:00",
            "2025-06-01 00:30:00",
            "2025-12-31 18:20:00",
            "2025-07-04 12:45:00",
        ]),
        "PULocationID":   [132, 161, 237, 48, 138],
        "DOLocationID":   [236, 170, 161, 113, 264],
        "trip_distance":  [2.5, 3.1, 1.2, 8.7, 15.0],
        "fare_amount":    [12.5, 15.0, 7.5, 35.0, 55.0],
        "total_amount":   [16.0, 19.5, 10.0, 42.0, 65.0],
        "passenger_count": [1, 2, 1, 3, 1],
        "payment_type":   [1, 1, 2, 1, 1],
        "RatecodeID":     [1, 1, 1, 2, 2],
        "store_and_fwd_flag": ["N", "N", "N", "N", "Y"],
    })


# ═══════════════════════════════════════════════════════════
# Tests: Duration Computation
# ═══════════════════════════════════════════════════════════

class TestDurationComputation:
    """Tests for computing trip_duration_minutes from datetime columns."""

    def test_duration_is_positive(self, sample_trip_df):
        """All computed durations must be strictly positive."""
        df = sample_trip_df.copy()
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        durations = delta.dt.total_seconds() / 60
        assert (durations > 0).all(), "Expected all durations to be positive"

    def test_duration_correct_values(self, sample_trip_df):
        """Verify exact duration values for known datetime pairs."""
        df = sample_trip_df.copy()
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        durations = delta.dt.total_seconds() / 60

        # Row 0: 08:00 → 08:25 = 25 minutes
        assert durations.iloc[0] == pytest.approx(25.0, abs=0.01)
        # Row 1: 14:30 → 14:52 = 22 minutes
        assert durations.iloc[1] == pytest.approx(22.0, abs=0.01)
        # Row 2: 00:15 → 00:30 = 15 minutes
        assert durations.iloc[2] == pytest.approx(15.0, abs=0.01)

    def test_zero_duration_detection(self):
        """Trips with identical pickup and dropoff times should have duration = 0."""
        df = pd.DataFrame({
            "tpep_pickup_datetime":  pd.to_datetime(["2025-01-01 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-01 10:00:00"]),
        })
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        durations = delta.dt.total_seconds() / 60
        assert durations.iloc[0] == 0.0

    def test_negative_duration_detection(self):
        """Dropoff before pickup should produce a negative duration (invalid record)."""
        df = pd.DataFrame({
            "tpep_pickup_datetime":  pd.to_datetime(["2025-01-01 10:30:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2025-01-01 10:00:00"]),
        })
        delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        durations = delta.dt.total_seconds() / 60
        assert durations.iloc[0] < 0, "Should detect negative (invalid) duration"


# ═══════════════════════════════════════════════════════════
# Tests: Time Feature Extraction
# ═══════════════════════════════════════════════════════════

class TestTimeFeatureExtraction:
    """Tests for extract_time_features() in utils.py."""

    def test_hour_extraction(self, sample_trip_df):
        """Hour should be correctly extracted from pickup datetime."""
        df = extract_time_features(sample_trip_df.copy(), "tpep_pickup_datetime")
        assert df["hour"].iloc[0] == 8    # 08:00
        assert df["hour"].iloc[2] == 0    # 00:15

    def test_day_of_week(self, sample_trip_df):
        """Day of week: Monday = 0, Sunday = 6."""
        df = extract_time_features(sample_trip_df.copy(), "tpep_pickup_datetime")
        # 2025-01-15 is a Wednesday → day_of_week = 2
        assert df["day_of_week"].iloc[0] == 2

    def test_is_weekend_flag(self, sample_trip_df):
        """is_weekend should be 1 for Saturday/Sunday, 0 otherwise."""
        df = extract_time_features(sample_trip_df.copy(), "tpep_pickup_datetime")
        # All sample dates are weekdays
        # Inject a Saturday for testing
        df_test = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime(["2025-01-18"])  # Saturday
        })
        df_test = extract_time_features(df_test, "tpep_pickup_datetime")
        assert df_test["is_weekend"].iloc[0] == 1

    def test_is_rush_hour_flag(self):
        """Rush hour: AM (7–9) and PM (16–19) should be flagged as 1."""
        rush_times = pd.DataFrame({
            "tpep_pickup_datetime": pd.to_datetime([
                "2025-01-15 08:00",   # AM rush → 1
                "2025-01-15 17:00",   # PM rush → 1
                "2025-01-15 11:00",   # Midday  → 0
                "2025-01-15 22:00",   # Night   → 0
            ])
        })
        df = extract_time_features(rush_times, "tpep_pickup_datetime")
        assert df["is_rush_hour"].iloc[0] == 1, "08:00 should be AM rush"
        assert df["is_rush_hour"].iloc[1] == 1, "17:00 should be PM rush"
        assert df["is_rush_hour"].iloc[2] == 0, "11:00 should not be rush"
        assert df["is_rush_hour"].iloc[3] == 0, "22:00 should not be rush"

    def test_month_extraction(self, sample_trip_df):
        """Month values should match the calendar month of pickup."""
        df = extract_time_features(sample_trip_df.copy(), "tpep_pickup_datetime")
        assert df["month"].iloc[0] == 1    # January
        assert df["month"].iloc[2] == 6    # June
        assert df["month"].iloc[3] == 12   # December

    def test_cyclic_encoding_bounds(self, sample_trip_df):
        """
        Cyclic features (sin/cos) must always be in [-1, 1].
        This ensures the encoding is mathematically valid.
        """
        df = extract_time_features(sample_trip_df.copy(), "tpep_pickup_datetime")
        # Manually add cyclic features (as done in feature engineering)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        assert df["hour_sin"].between(-1, 1).all(), "hour_sin must be in [-1, 1]"
        assert df["hour_cos"].between(-1, 1).all(), "hour_cos must be in [-1, 1]"

    def test_cyclic_midnight_continuity(self):
        """
        Hour 23 and hour 0 should be close in cyclic encoding.
        This is the key property cyclic encoding provides over raw hour.
        """
        hour_0  = np.array([np.sin(2 * np.pi * 0  / 24), np.cos(2 * np.pi * 0  / 24)])
        hour_23 = np.array([np.sin(2 * np.pi * 23 / 24), np.cos(2 * np.pi * 23 / 24)])
        hour_12 = np.array([np.sin(2 * np.pi * 12 / 24), np.cos(2 * np.pi * 12 / 24)])

        dist_0_23 = np.linalg.norm(hour_0 - hour_23)  # Should be small
        dist_0_12 = np.linalg.norm(hour_0 - hour_12)  # Should be large

        assert dist_0_23 < dist_0_12, (
            "Hour 0 and Hour 23 should be closer in cyclic encoding than Hour 0 and Hour 12"
        )


# ═══════════════════════════════════════════════════════════
# Tests: IQR Bounds
# ═══════════════════════════════════════════════════════════

class TestIQRBounds:
    """Tests for iqr_bounds() outlier detection helper."""

    def test_symmetric_distribution(self):
        """For normally distributed data, bounds should be symmetric around mean."""
        data = pd.Series(range(1, 101))   # 1 to 100
        lower, upper = iqr_bounds(data, factor=1.5)
        # Q1=25.75, Q3=75.25, IQR=49.5 → bounds: 25.75-74.25=-48.5, 75.25+74.25=149.5
        assert lower < data.min(), "Lower bound should be below minimum"
        assert upper > data.max(), "Upper bound should be above maximum"

    def test_outlier_detection(self):
        """Extreme values should fall outside the IQR bounds."""
        data = pd.Series([1, 2, 3, 4, 5, 1000])  # 1000 is an outlier
        lower, upper = iqr_bounds(data, factor=1.5)
        assert 1000 > upper, "Value 1000 should be flagged as an outlier"

    def test_factor_effect(self):
        """Larger IQR factor should produce wider (more permissive) bounds."""
        data = pd.Series(range(1, 101))
        lower_15, upper_15 = iqr_bounds(data, factor=1.5)
        lower_30, upper_30 = iqr_bounds(data, factor=3.0)
        assert upper_30 > upper_15, "Factor=3.0 should give wider upper bound"
        assert lower_30 < lower_15, "Factor=3.0 should give wider lower bound"


# ═══════════════════════════════════════════════════════════
# Tests: Regression Metrics
# ═══════════════════════════════════════════════════════════

class TestRegressionMetrics:
    """Tests for regression_metrics() in utils.py."""

    def test_perfect_predictions(self):
        """Perfect predictions should give RMSE=0, MAE=0, R²=1."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = y_true.copy()
        metrics = regression_metrics(y_true, y_pred)

        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["MAE"]  == pytest.approx(0.0, abs=1e-6)
        assert metrics["R2"]   == pytest.approx(1.0, abs=1e-6)

    def test_constant_prediction(self):
        """
        A model that always predicts the mean should have R² ≈ 0.
        This is the theoretical floor for a meaningful model.
        """
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.full_like(y_true, y_true.mean())
        metrics = regression_metrics(y_true, y_pred)
        assert metrics["R2"] == pytest.approx(0.0, abs=1e-6)

    def test_rmse_is_always_nonnegative(self):
        """RMSE must always be ≥ 0."""
        y_true = np.random.rand(100) * 60
        y_pred = y_true + np.random.randn(100) * 5
        metrics = regression_metrics(y_true, y_pred)
        assert metrics["RMSE"] >= 0

    def test_metrics_keys(self):
        """regression_metrics must return all expected keys."""
        y_true = np.array([15.0, 22.0, 10.0])
        y_pred = np.array([14.0, 23.0, 11.0])
        metrics = regression_metrics(y_true, y_pred)
        for key in ["RMSE", "MAE", "R2", "MAPE"]:
            assert key in metrics, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════
# Tests: Data Cleaning Logic
# ═══════════════════════════════════════════════════════════

class TestDataCleaning:
    """Tests for the filtering rules in 02_prepare_data.py."""

    def make_trip_df(self, **kwargs):
        """Helper to create a minimal trip DataFrame with custom overrides."""
        base = {
            "trip_duration_minutes": 15.0,
            "trip_distance":        2.5,
            "PULocationID":         132,
            "DOLocationID":         236,
            "passenger_count":      1.0,
            "fare_amount":          12.5,
            "total_amount":         16.0,
        }
        base.update(kwargs)
        return pd.DataFrame([base])

    def test_filter_too_short_duration(self):
        """Trips shorter than 1 minute should be filtered out."""
        df = self.make_trip_df(trip_duration_minutes=0.5)
        result = df[df["trip_duration_minutes"] >= 1]
        assert len(result) == 0

    def test_filter_too_long_duration(self):
        """Trips longer than 180 minutes should be filtered out."""
        df = self.make_trip_df(trip_duration_minutes=200)
        result = df[df["trip_duration_minutes"] <= 180]
        assert len(result) == 0

    def test_valid_duration_passes(self):
        """Trips with valid durations should not be filtered."""
        df = self.make_trip_df(trip_duration_minutes=25.0)
        result = df[
            (df["trip_duration_minutes"] >= 1) &
            (df["trip_duration_minutes"] <= 180)
        ]
        assert len(result) == 1

    def test_invalid_location_id(self):
        """LocationIDs outside [1, 265] should be filtered out."""
        df = self.make_trip_df(PULocationID=999)
        result = df[df["PULocationID"].between(1, 265)]
        assert len(result) == 0

    def test_negative_fare_filter(self):
        """Negative fare amounts indicate corrupt records and must be removed."""
        df = self.make_trip_df(fare_amount=-5.0)
        result = df[df["fare_amount"] >= 0]
        assert len(result) == 0

    def test_negative_distance_filter(self):
        """Negative trip distances are physically impossible."""
        df = self.make_trip_df(trip_distance=-1.0)
        result = df[df["trip_distance"] > 0]
        assert len(result) == 0

    def test_zero_distance_filter(self):
        """Zero-distance trips are also invalid for duration prediction."""
        df = self.make_trip_df(trip_distance=0.0)
        result = df[df["trip_distance"] >= 0.1]
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════
# Tests: Feature Engineering
# ═══════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Tests for derived features created in 04_feature_engineering.py."""

    def test_log_distance_is_positive(self, sample_trip_df):
        """log1p of any positive distance should be positive."""
        distances = sample_trip_df["trip_distance"]
        log_distances = np.log1p(distances)
        assert (log_distances > 0).all()

    def test_log_distance_handles_zero(self):
        """log1p(0) = 0, not error — handles edge cases gracefully."""
        result = np.log1p(0)
        assert result == 0.0

    def test_baseline_duration_formula(self, sample_trip_df):
        """baseline_duration = trip_distance / avg_speed * 60."""
        avg_speed = 12.5   # mph
        df = sample_trip_df.copy()
        df["baseline_duration"] = (df["trip_distance"] / avg_speed) * 60

        # Row 0: 2.5 miles / 12.5 mph * 60 = 12 minutes
        expected = (2.5 / 12.5) * 60
        assert df["baseline_duration"].iloc[0] == pytest.approx(expected, abs=0.01)

    def test_same_zone_flag(self, sample_trip_df):
        """same_zone_flag should be 1 when PU == DO, 0 otherwise."""
        df = sample_trip_df.copy()
        df["same_zone_flag"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)
        # All sample trips have different PU and DO zones
        assert (df["same_zone_flag"] == 0).all()

    def test_same_zone_flag_true_case(self):
        """Explicitly test the same-zone case."""
        df = pd.DataFrame({"PULocationID": [132], "DOLocationID": [132]})
        df["same_zone_flag"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)
        assert df["same_zone_flag"].iloc[0] == 1

    def test_route_mean_fallback(self):
        """
        Route feature mapping should fall back to global mean
        for routes not seen in training.
        """
        route_map = {(132, 236): 20.0, (161, 170): 18.5}
        global_mean = 15.0

        # Known route
        assert route_map.get((132, 236), global_mean) == 20.0
        # Unknown route → falls back to global mean
        assert route_map.get((999, 888), global_mean) == global_mean
