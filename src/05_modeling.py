# ============================================================
# src/05_modeling.py
# Phase 5: Model Training & Comparison
#
# PURPOSE:
#   Train, tune, and compare four models against the baseline.
#   All training uses 2025 data only. The 2026 test set is
#   kept sealed until evaluation in 06_evaluation.py.
#
# MODELS:
#   0. Baseline          — distance ÷ average_speed (no ML)
#   1. Linear Regression — interpretable starting point
#   2. Ridge Regression  — regularized linear (handles multicollinearity)
#   3. LightGBM          — primary production model (gradient boosting)
#   4. MLP Neural Network— 4th model requirement
#
# VALIDATION STRATEGY:
#   Time-series cross-validation (TimeSeriesSplit) — NEVER random shuffle.
#   The data has temporal ordering: we always train on past, validate on future.
#   n_splits=5 means: train on month 1–2, val month 3 → ... → train 1–9, val 10+
#
# HYPERPARAMETER TUNING:
#   Ridge  → GridSearchCV over alpha values
#   LightGBM → RandomizedSearchCV over key hyperparameters
#
# HOW TO RUN:
#   python src/05_modeling.py
#
# OUTPUT:
#   models/linear_regression.joblib
#   models/ridge_regression.joblib
#   models/lightgbm.joblib
#   models/mlp.joblib
#   models/scaler.joblib            (StandardScaler for linear models)
#   models/validation_scores.json  (CV scores for all models)
# ============================================================

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_logger, get_data_path, get_models_path,
    print_section, regression_metrics, print_metrics
)

logger = get_logger("05_modeling")

TARGET = "trip_duration_minutes"
RANDOM_STATE = 42

# ── Load Features ─────────────────────────────────────────────

def load_features():
    """
    Load the engineered training feature matrix.
    Also load the feature column list so we always use
    the exact same set of features across all models.

    Returns:
        X_train (DataFrame), y_train (Series), feature_cols (list)
    """
    processed_dir = get_data_path("processed")

    # Load feature column list
    feat_path = os.path.join(processed_dir, "feature_columns.txt")
    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f.readlines()]

    # Load training data
    train_path = os.path.join(processed_dir, "train_features.parquet")
    df = pd.read_parquet(train_path)

    logger.info(f"  Training data shape: {df.shape}")
    logger.info(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].astype("float32")
    y = df[TARGET].astype("float32")

    return X, y, feature_cols


# ── Validation Scorer ─────────────────────────────────────────

def time_series_cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> dict:
    """
    Evaluate a model using TimeSeriesSplit cross-validation.

    WHY TimeSeriesSplit?
        Regular k-fold randomly shuffles rows, which means a model
        can train on August data and validate on January data. This
        is data leakage — the model would never see future data in
        production. TimeSeriesSplit always validates on a period
        AFTER all training data, mimicking real deployment.

    Args:
        model:    Scikit-learn compatible estimator
        X:        Feature matrix
        y:        Target series
        n_splits: Number of CV folds (default 5)

    Returns:
        Dictionary of mean ± std for each metric
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores, mae_scores, r2_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        # Clip negative predictions (duration can't be negative)
        preds = np.clip(preds, 0, None)

        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_scores.append(mean_absolute_error(y_val, preds))
        r2_scores.append(r2_score(y_val, preds))

        logger.info(
            f"    Fold {fold+1}: RMSE={rmse_scores[-1]:.3f}  "
            f"MAE={mae_scores[-1]:.3f}  R²={r2_scores[-1]:.4f}"
        )

    return {
        "RMSE_mean": round(float(np.mean(rmse_scores)), 4),
        "RMSE_std":  round(float(np.std(rmse_scores)),  4),
        "MAE_mean":  round(float(np.mean(mae_scores)),  4),
        "MAE_std":   round(float(np.std(mae_scores)),   4),
        "R2_mean":   round(float(np.mean(r2_scores)),   4),
        "R2_std":    round(float(np.std(r2_scores)),    4),
    }


# ── Model 0: Baseline ─────────────────────────────────────────

def evaluate_baseline(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the naive distance÷speed baseline on training data.

    The baseline doesn't 'learn' anything — it simply converts distance
    to duration using a fixed speed. If our ML models can't beat this,
    they're not adding value.

    Formula:  baseline_duration = trip_distance / avg_speed * 60
    (This value was precomputed in feature engineering as 'baseline_duration')

    Args:
        X: Feature matrix (must contain 'baseline_duration' column)
        y: Actual durations

    Returns:
        Dictionary of baseline metrics
    """
    print_section("Model 0: Baseline (Distance ÷ Speed)")

    preds = X["baseline_duration"].values
    metrics = regression_metrics(y.values, preds)
    print_metrics(metrics, "Baseline")

    return {"baseline": metrics}


# ── Model 1: Linear Regression ────────────────────────────────

def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a simple Linear Regression model.

    Why Linear Regression first?
        It's fully interpretable and gives us a sanity check —
        if tree models don't beat this significantly, we may have
        a feature engineering problem, not a model problem.

    Note: We scale features before fitting linear models because
    regularization and gradient-based optimization are sensitive
    to feature magnitude differences.

    Args:
        X: Feature matrix
        y: Target series

    Returns:
        (fitted_pipeline, cv_scores)
    """
    print_section("Model 1: Linear Regression")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression(n_jobs=-1))
    ])

    logger.info("  Running TimeSeriesSplit cross-validation...")
    start = time.time()
    cv_scores = time_series_cv_score(pipeline, X, y, n_splits=5)
    elapsed = time.time() - start

    logger.info(f"  CV completed in {elapsed:.1f}s")
    print_metrics(
        {"RMSE": cv_scores["RMSE_mean"], "MAE": cv_scores["MAE_mean"],
         "R2": cv_scores["R2_mean"], "MAPE": 0},
        "Linear Regression (CV)"
    )

    # Final fit on all training data
    pipeline.fit(X, y)
    return pipeline, cv_scores


# ── Model 2: Ridge Regression ─────────────────────────────────

def train_ridge_regression(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train Ridge Regression with hyperparameter tuning.

    Why Ridge over plain Linear Regression?
        Ridge adds an L2 penalty term (α × sum(w²)) to the loss function.
        This shrinks large coefficients, reducing overfitting caused by
        multicollinearity (e.g., trip_distance and baseline_duration
        are highly correlated).

    Tuning:
        We search for the best regularization strength 'alpha'
        using GridSearchCV with TimeSeriesSplit.

    Args:
        X: Feature matrix
        y: Target series

    Returns:
        (best_fitted_pipeline, cv_scores)
    """
    print_section("Model 2: Ridge Regression (with hyperparameter tuning)")

    # Alpha search space: from very weak to very strong regularization
    alpha_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge())
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    logger.info(f"  Searching alpha grid: {alpha_grid['model__alpha']}")
    start = time.time()

    grid_search = GridSearchCV(
        pipeline,
        param_grid=alpha_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X, y)
    elapsed = time.time() - start

    best_alpha = grid_search.best_params_["model__alpha"]
    best_rmse  = -grid_search.best_score_

    logger.info(f"  Grid search completed in {elapsed:.1f}s")
    logger.info(f"  Best alpha: {best_alpha}  (CV RMSE: {best_rmse:.4f} min)")

    # Recompute full CV scores with the best estimator
    cv_scores = time_series_cv_score(grid_search.best_estimator_, X, y, n_splits=5)
    cv_scores["best_alpha"] = best_alpha

    print_metrics(
        {"RMSE": cv_scores["RMSE_mean"], "MAE": cv_scores["MAE_mean"],
         "R2": cv_scores["R2_mean"], "MAPE": 0},
        f"Ridge Regression (alpha={best_alpha}, CV)"
    )

    return grid_search.best_estimator_, cv_scores


# ── Model 3: LightGBM ─────────────────────────────────────────

def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a LightGBM gradient-boosting model with hyperparameter tuning.

    Why LightGBM?
        - Handles non-linear relationships (traffic isn't linear)
        - Native support for high-cardinality categorical features
        - Much faster training than XGBoost on large datasets
        - Typically best performance on tabular data

    Tuning:
        RandomizedSearchCV is used instead of GridSearchCV because
        the LightGBM hyperparameter space is large. Random search
        finds good solutions faster than exhaustive grid search.

    Key hyperparameters:
        num_leaves:     Complexity of each tree (higher = more complex)
        learning_rate:  Step size for gradient descent (lower = more robust)
        n_estimators:   Number of trees (with early stopping)
        max_depth:      Max tree depth (-1 = unlimited)
        min_child_samples: Minimum samples per leaf (prevents overfitting)
        subsample:      Fraction of rows used per tree (regularization)
        colsample_bytree: Fraction of features used per tree

    Args:
        X: Feature matrix
        y: Target series

    Returns:
        (best_fitted_model, cv_scores)
    """
    print_section("Model 3: LightGBM (with hyperparameter tuning)")

    param_distributions = {
        "num_leaves":        [31, 63, 127, 255],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "n_estimators":      [200, 500, 1000],
        "max_depth":         [-1, 6, 10, 15],
        "min_child_samples": [20, 50, 100],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
        "reg_alpha":         [0.0, 0.1, 0.5],   # L1 regularization
        "reg_lambda":        [0.0, 0.1, 1.0],   # L2 regularization
    }

    lgbm_base = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1          # Silence LightGBM's per-iteration output
    )

    tscv = TimeSeriesSplit(n_splits=5)

    logger.info("  Running RandomizedSearchCV (n_iter=30)...")
    logger.info("  This may take several minutes on the full dataset...")
    start = time.time()

    random_search = RandomizedSearchCV(
        lgbm_base,
        param_distributions=param_distributions,
        n_iter=30,              # Try 30 random combinations
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    random_search.fit(X, y)
    elapsed = time.time() - start

    best_params = random_search.best_params_
    best_rmse   = -random_search.best_score_

    logger.info(f"  Random search completed in {elapsed:.1f}s")
    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Best CV RMSE: {best_rmse:.4f} min")

    # Full CV evaluation with best model
    cv_scores = time_series_cv_score(random_search.best_estimator_, X, y, n_splits=5)
    cv_scores["best_params"] = best_params

    print_metrics(
        {"RMSE": cv_scores["RMSE_mean"], "MAE": cv_scores["MAE_mean"],
         "R2": cv_scores["R2_mean"], "MAPE": 0},
        "LightGBM (CV)"
    )

    return random_search.best_estimator_, cv_scores


# ── Model 4: MLP Neural Network ───────────────────────────────

def train_mlp(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a Multi-Layer Perceptron (MLP) Neural Network.

    Architecture:
        Input → Dense(256) → ReLU → Dense(128) → ReLU → Dense(64) → ReLU → Output(1)

    Why MLP?
        Neural networks can theoretically learn any function.
        On large tabular datasets, MLPs sometimes capture complex
        interaction patterns that linear models miss, though they
        typically don't outperform tree-based models on tabular data.

    Notes:
        - We use StandardScaler because neural networks are very
          sensitive to feature scale
        - early_stopping=True prevents overfitting by monitoring
          a validation holdout during training
        - max_iter is set high — early stopping will kick in before

    Args:
        X: Feature matrix
        y: Target series

    Returns:
        (fitted_pipeline, cv_scores)
    """
    print_section("Model 4: MLP Neural Network")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,            # L2 regularization strength
            batch_size=2048,        # Mini-batch size
            learning_rate="adaptive",
            max_iter=200,
            early_stopping=True,    # Stop if validation loss doesn't improve
            validation_fraction=0.1,
            n_iter_no_change=15,    # Stop after 15 rounds of no improvement
            random_state=RANDOM_STATE,
            verbose=False
        ))
    ])

    logger.info("  Running TimeSeriesSplit cross-validation...")
    logger.info("  Note: MLP training may take longer than tree-based models...")
    start = time.time()
    cv_scores = time_series_cv_score(pipeline, X, y, n_splits=3)  # 3 folds for speed
    elapsed = time.time() - start

    logger.info(f"  CV completed in {elapsed:.1f}s")
    print_metrics(
        {"RMSE": cv_scores["RMSE_mean"], "MAE": cv_scores["MAE_mean"],
         "R2": cv_scores["R2_mean"], "MAPE": 0},
        "MLP Neural Network (CV)"
    )

    # Final fit
    pipeline.fit(X, y)
    return pipeline, cv_scores


# ── Save Model ────────────────────────────────────────────────

def save_model(model, name: str) -> None:
    """Save a fitted model to models/ directory using joblib."""
    models_dir = get_models_path()
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"  ✓ Model saved: {path} ({size_mb:.2f} MB)")


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 5: MODELING")
    print("="*60)

    # ── Load data
    X_train, y_train, feature_cols = load_features()
    logger.info(f"  Loaded training data: {X_train.shape}")

    all_scores = {}

    # ── Baseline (no training needed)
    base_scores = evaluate_baseline(X_train, y_train)
    all_scores.update(base_scores)

    # ── Model 1: Linear Regression
    lr_model, lr_scores = train_linear_regression(X_train, y_train)
    save_model(lr_model, "linear_regression")
    all_scores["linear_regression"] = lr_scores

    # ── Model 2: Ridge Regression
    ridge_model, ridge_scores = train_ridge_regression(X_train, y_train)
    save_model(ridge_model, "ridge_regression")
    all_scores["ridge_regression"] = ridge_scores

    # ── Model 3: LightGBM
    lgbm_model, lgbm_scores = train_lightgbm(X_train, y_train)
    save_model(lgbm_model, "lightgbm")
    all_scores["lightgbm"] = lgbm_scores

    # ── Model 4: MLP
    mlp_model, mlp_scores = train_mlp(X_train, y_train)
    save_model(mlp_model, "mlp")
    all_scores["mlp"] = mlp_scores

    # ── Save all CV scores as JSON for easy reference
    scores_path = os.path.join(get_models_path(), "validation_scores.json")
    with open(scores_path, "w") as f:
        json.dump(all_scores, f, indent=2, default=str)
    logger.info(f"\n  ✓ All validation scores saved: {scores_path}")

    # ── Print Summary Table
    print_section("Cross-Validation Summary")
    print(f"\n  {'Model':<25} {'CV RMSE':>10} {'CV MAE':>10} {'CV R²':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Baseline':<25} {all_scores['baseline']['RMSE']:>10.4f} "
          f"{all_scores['baseline']['MAE']:>10.4f} "
          f"{all_scores['baseline']['R2']:>10.4f}")

    for name in ["linear_regression", "ridge_regression", "lightgbm", "mlp"]:
        s = all_scores[name]
        print(f"  {name:<25} {s['RMSE_mean']:>10.4f} {s['MAE_mean']:>10.4f} {s['R2_mean']:>10.4f}")

    print("\n✅ Phase 5 complete. Models saved to models/")
    print("   Next step: python src/06_evaluation.py\n")
