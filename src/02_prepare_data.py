# ============================================================
# src/02_prepare_data.py
# Phase 2: Data Preparation & Cleaning
#
# PURPOSE:
#   Load all raw parquet files, clean the data, remove outliers,
#   optimize memory usage, and save one clean processed file.
#
# WHAT WE DO:
#   1. Load all 14 monthly parquet files into one DataFrame
#   2. Compute the target variable: trip_duration_minutes
#   3. Remove invalid, corrupt, and extreme-outlier records
#   4. Optimize column data types to reduce memory usage
#   5. Save cleaned data to data/processed/
#
# HOW TO RUN:
#   python src/02_prepare_data.py
#
# OUTPUT:
#   data/processed/trips_2025_clean.parquet   (2025 training set)
#   data/processed/trips_2026_clean.parquet   (2026 test set)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_logger, get_data_path, print_section,
    validate_dataframe, iqr_bounds
)

logger = get_logger("02_prepare_data")

# ── Constants ────────────────────────────────────────────────

# Columns we actually need — dropping unused ones saves RAM
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
    "payment_type",
    "RatecodeID",
    "store_and_fwd_flag",
]

# Duration business rules (in minutes)
MIN_DURATION_MINUTES = 1      # Trips under 1 min are almost certainly errors
MAX_DURATION_MINUTES = 180    # Trips over 3 hours are extreme outliers

# Distance business rules (in miles)
MIN_DISTANCE_MILES = 0.1      # Less than ~500 feet — usually errors
MAX_DISTANCE_MILES = 100      # NYC taxi trips won't go farther than this

# Valid LocationID range per TLC data dictionary
VALID_LOCATION_MIN = 1
VALID_LOCATION_MAX = 265


# ── Load Data ────────────────────────────────────────────────

def load_parquet_files(year: int) -> pd.DataFrame:
    """
    Load all monthly parquet files for a given year and
    combine them into a single DataFrame.

    For 2026, we only load January and February.

    Args:
        year: 4-digit year (2025 or 2026)

    Returns:
        Combined DataFrame for that year
    """
    raw_dir = get_data_path("raw")
    months = range(1, 13) if year == 2025 else range(1, 3)

    frames = []
    total_rows = 0

    for month in months:
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join(raw_dir, filename)

        if not os.path.exists(filepath):
            logger.warning(f"  File not found, skipping: {filename}")
            continue

        logger.info(f"  Loading: {filename}")

        # Read only the columns we need — much faster and lighter
        df_month = pd.read_parquet(filepath, columns=REQUIRED_COLUMNS)
        n = len(df_month)
        total_rows += n
        logger.info(f"    → {n:,} rows loaded")
        frames.append(df_month)

    if not frames:
        raise FileNotFoundError(
            f"No parquet files found for {year}. "
            "Run 01_download_data.py first."
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"\n  Total rows loaded for {year}: {total_rows:,}")
    return df


# ── Compute Target Variable ───────────────────────────────────

def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trip_duration_minutes — our prediction target.

    Formula:
        duration = (dropoff_datetime - pickup_datetime).total_seconds() / 60

    We keep the raw datetime columns for feature engineering later.

    Args:
        df: Raw trip DataFrame

    Returns:
        DataFrame with 'trip_duration_minutes' column added
    """
    logger.info("  Computing trip_duration_minutes...")

    # Ensure both columns are parsed as datetime
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Calculate duration in minutes
    delta = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["trip_duration_minutes"] = delta.dt.total_seconds() / 60

    logger.info(f"  Duration stats (before cleaning):")
    logger.info(f"    Min:    {df['trip_duration_minutes'].min():.2f} min")
    logger.info(f"    Max:    {df['trip_duration_minutes'].max():.2f} min")
    logger.info(f"    Mean:   {df['trip_duration_minutes'].mean():.2f} min")
    logger.info(f"    Median: {df['trip_duration_minutes'].median():.2f} min")

    return df


# ── Cleaning Pipeline ─────────────────────────────────────────

def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a multi-step cleaning pipeline to remove corrupt,
    invalid, and extreme-outlier records.

    Each filter is logged so we know exactly how many rows
    each rule removes — important for documentation.

    Filters applied (in order):
        1. Negative or zero duration
        2. Duration outside [1, 180] minutes
        3. Negative or zero trip distance
        4. Distance outside [0.1, 100] miles
        5. Invalid pickup / dropoff location IDs
        6. Invalid passenger count (0 or > 8)
        7. Negative fare amounts
        8. Duplicate rows
        9. Records with wrong year (TLC sometimes has data errors)

    Args:
        df: DataFrame after duration computation

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    logger.info(f"\n  Starting cleaning. Initial rows: {initial_count:,}")

    def log_filter(df_before, df_after, rule_name):
        removed = len(df_before) - len(df_after)
        pct = 100 * removed / initial_count
        logger.info(f"    [{rule_name}] Removed {removed:,} rows ({pct:.3f}%)")
        return df_after

    # ── Filter 1: Negative or zero duration ──────────────────
    df_f = df[df["trip_duration_minutes"] > 0]
    df = log_filter(df, df_f, "Negative duration")

    # ── Filter 2: Duration business rules ────────────────────
    df_f = df[
        (df["trip_duration_minutes"] >= MIN_DURATION_MINUTES) &
        (df["trip_duration_minutes"] <= MAX_DURATION_MINUTES)
    ]
    df = log_filter(df, df_f, "Duration [1–180 min]")

    # ── Filter 3: Negative or zero distance ──────────────────
    df_f = df[df["trip_distance"] > 0]
    df = log_filter(df, df_f, "Negative distance")

    # ── Filter 4: Distance business rules ────────────────────
    df_f = df[
        (df["trip_distance"] >= MIN_DISTANCE_MILES) &
        (df["trip_distance"] <= MAX_DISTANCE_MILES)
    ]
    df = log_filter(df, df_f, "Distance [0.1–100 mi]")

    # ── Filter 5: Valid LocationIDs ───────────────────────────
    df_f = df[
        df["PULocationID"].between(VALID_LOCATION_MIN, VALID_LOCATION_MAX) &
        df["DOLocationID"].between(VALID_LOCATION_MIN, VALID_LOCATION_MAX)
    ]
    df = log_filter(df, df_f, "Invalid LocationID")

    # ── Filter 6: Valid passenger count ──────────────────────
    # NaN passenger_count is acceptable — we'll fill it later
    df_f = df[
        df["passenger_count"].isna() |
        df["passenger_count"].between(1, 8)
    ]
    df = log_filter(df, df_f, "Passenger count [1–8]")

    # ── Filter 7: Non-negative fare ──────────────────────────
    df_f = df[df["fare_amount"] >= 0]
    df = log_filter(df, df_f, "Negative fare")

    # ── Filter 8: Exact duplicate rows ───────────────────────
    df_f = df.drop_duplicates()
    df = log_filter(df, df_f, "Duplicate rows")

    # ── Summary ───────────────────────────────────────────────
    final_count = len(df)
    total_removed = initial_count - final_count
    logger.info(
        f"\n  Cleaning complete."
        f"\n    Rows before: {initial_count:,}"
        f"\n    Rows after:  {final_count:,}"
        f"\n    Total removed: {total_removed:,} ({100*total_removed/initial_count:.2f}%)"
    )

    return df.reset_index(drop=True)


# ── Handle Missing Values ─────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop remaining missing values after filtering.

    Strategy per column:
        passenger_count  → Fill with median (common in TLC data)
        RatecodeID       → Fill with mode (most common rate code = 1)
        store_and_fwd_flag → Fill with 'N' (not stored and forwarded)
        payment_type     → Fill with mode

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with no missing values in used columns
    """
    logger.info("\n  Handling missing values...")

    before = df.isnull().sum().sum()

    # Numeric: fill with median
    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].median()
    )

    # Categorical-like: fill with mode
    df["RatecodeID"] = df["RatecodeID"].fillna(
        df["RatecodeID"].mode()[0]
    )
    df["payment_type"] = df["payment_type"].fillna(
        df["payment_type"].mode()[0]
    )

    # Flag column: default to 'N' (not stored/forwarded)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")

    after = df.isnull().sum().sum()
    logger.info(f"    Missing values reduced from {before:,} to {after:,}")

    return df


# ── Data Type Optimization ────────────────────────────────────

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory footprint.
    This is essential for 14 months of data (~30–50M rows total).

    Conversions:
        int64  → int32 or int16 (where value range allows)
        float64 → float32 (sufficient precision for our features)
        object  → category (for low-cardinality string columns)

    Args:
        df: DataFrame to optimize

    Returns:
        Memory-optimized DataFrame
    """
    logger.info("\n  Optimizing data types...")
    before_mb = df.memory_usage(deep=True).sum() / 1e6

    # LocationIDs: max value is 265 → int16 is enough (max 32,767)
    for col in ["PULocationID", "DOLocationID"]:
        df[col] = df[col].astype("int16")

    # Passenger count: max 8 → int8 is enough (max 127)
    df["passenger_count"] = df["passenger_count"].astype("float32")

    # Rate code and payment type: small integers
    df["RatecodeID"]   = df["RatecodeID"].astype("float32")
    df["payment_type"] = df["payment_type"].astype("float32")

    # Float columns: downcast from float64 → float32
    for col in ["trip_distance", "fare_amount", "total_amount",
                "trip_duration_minutes"]:
        df[col] = df[col].astype("float32")

    # String column: low cardinality → category saves memory
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    savings = before_mb - after_mb
    logger.info(
        f"    Memory: {before_mb:.1f} MB → {after_mb:.1f} MB "
        f"(saved {savings:.1f} MB, {100*savings/before_mb:.1f}%)"
    )

    return df


# ── Save Processed Data ───────────────────────────────────────

def save_processed(df: pd.DataFrame, filename: str) -> None:
    """
    Save the cleaned DataFrame to data/processed/ as parquet.
    Parquet is preferred over CSV because it:
      - Preserves data types
      - Compresses data efficiently
      - Reads much faster than CSV for large files

    Args:
        df:       DataFrame to save
        filename: Output filename (e.g., 'trips_2025_clean.parquet')
    """
    processed_dir = get_data_path("processed")
    os.makedirs(processed_dir, exist_ok=True)

    out_path = os.path.join(processed_dir, filename)
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = os.path.getsize(out_path) / 1e6
    logger.info(f"\n  ✓ Saved: {out_path} ({size_mb:.1f} MB)")


# ── Main Pipeline ─────────────────────────────────────────────

def prepare_year(year: int) -> None:
    """
    Run the full preparation pipeline for one year of data.

    Steps:
        1. Load all monthly parquet files for the year
        2. Compute target variable (trip_duration_minutes)
        3. Remove invalid records
        4. Handle missing values
        5. Optimize data types
        6. Validate and save

    Args:
        year: 2025 (training data) or 2026 (test data)
    """
    label = "2025 Training" if year == 2025 else "2026 Test"
    print_section(f"Preparing {label} Data (Year: {year})")

    # Step 1: Load
    df = load_parquet_files(year)
    validate_dataframe(df, f"Raw {year} data")

    # Step 2: Target variable
    df = compute_duration(df)

    # Step 3: Clean
    df = remove_invalid_records(df)

    # Step 4: Missing values
    df = handle_missing_values(df)

    # Step 5: Optimize types
    df = optimize_dtypes(df)

    # Step 6: Final validation
    validate_dataframe(df, f"Cleaned {year} data")

    # Step 7: Save
    out_file = f"trips_{year}_clean.parquet"
    save_processed(df, out_file)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 2: DATA PREPARATION")
    print("="*60)

    # Process training data (2025) and test data (2026) separately
    # This ensures NO leakage between train and test sets
    prepare_year(2025)
    prepare_year(2026)

    print("\n✅ Phase 2 complete. Check data/processed/ for clean files.")
    print("   Next step: python src/03_eda.py\n")
