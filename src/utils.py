# ============================================================
# src/utils.py
# Shared utility functions used across all pipeline scripts.
# These are reusable helpers — keep them clean and well-tested.
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ── Logging Setup ────────────────────────────────────────────
# We configure a single logger that all scripts import.
# Logs go to both the console AND a log file for traceability.

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create and return a named logger that writes to console + file.

    Args:
        name:    Logger name (usually __name__ of the calling script)
        log_dir: Directory where log files are saved

    Returns:
        Configured logging.Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = os.path.join(log_dir, f"{name}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ── Path Helpers ─────────────────────────────────────────────

def get_project_root() -> str:
    """
    Return the absolute path to the project root directory.
    Assumes utils.py lives in src/ which is inside the project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(subdir: str = "raw") -> str:
    """
    Return path to a data subdirectory (raw or processed).

    Args:
        subdir: 'raw' or 'processed'

    Returns:
        Absolute path string
    """
    return os.path.join(get_project_root(), "data", subdir)


def get_models_path() -> str:
    """Return path to the models/ directory."""
    return os.path.join(get_project_root(), "models")


def get_figures_path() -> str:
    """Return path to the reports/figures/ directory."""
    return os.path.join(get_project_root(), "reports", "figures")


# ── Data Validation Helpers ───────────────────────────────────

def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a quick summary of a DataFrame for inspection.
    Useful after loading or transforming data.

    Args:
        df:   DataFrame to inspect
        name: Label to display in the output
    """
    print(f"\n{'='*55}")
    print(f"  Validating: {name}")
    print(f"{'='*55}")
    print(f"  Shape:        {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Columns:      {list(df.columns)}")
    print(f"\n  Missing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    ✓ No missing values found.")
    else:
        for col, cnt in missing.items():
            pct = 100 * cnt / len(df)
            print(f"    {col}: {cnt:,} missing ({pct:.2f}%)")
    print(f"{'='*55}\n")


def print_section(title: str) -> None:
    """Print a clearly visible section header to the console."""
    print(f"\n{'─'*60}")
    print(f"  {title.upper()}")
    print(f"{'─'*60}")


# ── Time Helpers ──────────────────────────────────────────────

def extract_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    Extract a full suite of time-based features from a datetime column.
    These are critical for capturing traffic patterns.

    Features created:
        hour, day_of_week, day_of_month, month, year,
        is_weekend, is_rush_hour, quarter, week_of_year

    Args:
        df:     DataFrame containing the datetime column
        dt_col: Name of the datetime column to extract from

    Returns:
        DataFrame with new time feature columns added
    """
    dt = pd.to_datetime(df[dt_col])

    df["hour"]         = dt.dt.hour
    df["day_of_week"]  = dt.dt.dayofweek        # 0=Monday, 6=Sunday
    df["day_of_month"] = dt.dt.day
    df["month"]        = dt.dt.month
    df["year"]         = dt.dt.year
    df["quarter"]      = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    # Weekend flag: Saturday (5) or Sunday (6)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Rush hour flag: AM rush 7–9, PM rush 16–19
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0
    )

    # Night flag: late night / early morning trips
    df["is_night"]     = df["hour"].apply(
        lambda h: 1 if (h >= 22) or (h <= 5) else 0
    )

    return df


# ── Outlier Helpers ───────────────────────────────────────────

def iqr_bounds(series: pd.Series, factor: float = 1.5):
    """
    Calculate IQR-based lower and upper bounds for outlier detection.

    Args:
        series: Numeric pandas Series
        factor: Multiplier for IQR (1.5 = standard, 3.0 = conservative)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - factor * IQR, Q3 + factor * IQR


# ── Metric Helpers ────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a full set of regression evaluation metrics.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R², MAPE
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # MAPE: avoid division by zero by filtering out zero actuals
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae, 4),
        "R2":   round(r2, 4),
        "MAPE": round(mape, 4)
    }


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n  📊 {model_name} Performance:")
    print(f"    RMSE : {metrics['RMSE']:.4f} minutes")
    print(f"    MAE  : {metrics['MAE']:.4f} minutes")
    print(f"    R²   : {metrics['R2']:.4f}")
    print(f"    MAPE : {metrics['MAPE']:.2f}%\n")
