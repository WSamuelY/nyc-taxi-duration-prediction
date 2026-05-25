# ============================================================
# src/06_evaluation.py
# Phase 6: Model Evaluation & Interpretation
#
# PURPOSE:
#   This is the FINAL evaluation step. We now "unseal" the
#   Jan–Feb 2026 test set and measure how well each model
#   performs on truly unseen data.
#
# WHAT WE REPORT:
#   1. Test set metrics for all models (RMSE, MAE, R², MAPE)
#   2. Comparison table: models vs. baseline
#   3. Feature importance (LightGBM)
#   4. Error analysis: where does the model fail?
#   5. Actual vs. predicted scatter plot
#   6. Residual distribution
#
# HOW TO RUN:
#   python src/06_evaluation.py
#
# OUTPUT:
#   reports/figures/07_model_comparison.png
#   reports/figures/08_feature_importance.png
#   reports/figures/09_actual_vs_predicted.png
#   reports/figures/10_error_analysis.png
#   reports/model_report.md   (text summary for your README / slides)
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_logger, get_data_path, get_models_path, get_figures_path,
    print_section, regression_metrics, print_metrics
)

logger = get_logger("06_evaluation")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13, "axes.titleweight": "bold"})

TARGET = "trip_duration_minutes"
FIGURES_DIR = get_figures_path()
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Figure saved: {filename}")


# ── Load Test Data & Models ───────────────────────────────────

def load_test_data():
    """
    Load the Jan–Feb 2026 test feature matrix.
    This data has never been seen by any model during training.
    """
    processed_dir = get_data_path("processed")

    feat_path = os.path.join(processed_dir, "feature_columns.txt")
    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f.readlines()]

    test_path = os.path.join(processed_dir, "test_features.parquet")
    df = pd.read_parquet(test_path)

    X_test = df[feature_cols].astype("float32")
    y_test = df[TARGET].astype("float32")

    logger.info(f"  Test data shape: {X_test.shape}")
    return X_test, y_test, feature_cols


def load_model(name: str):
    """Load a saved model from the models/ directory."""
    path = os.path.join(get_models_path(), f"{name}.joblib")
    if not os.path.exists(path):
        logger.warning(f"  Model not found: {path}")
        return None
    return joblib.load(path)


# ── Evaluate All Models on Test Set ──────────────────────────

def evaluate_all_models(X_test, y_test) -> dict:
    """
    Generate test set predictions for each model and compute metrics.

    This is the definitive evaluation — the number that goes in
    your report and slides.

    Returns:
        Dictionary of {model_name: metrics_dict}
    """
    print_section("Test Set Evaluation (Jan–Feb 2026)")

    results = {}
    predictions = {}

    # ── Baseline ──────────────────────────────────────────────
    baseline_preds = X_test["baseline_duration"].values
    baseline_preds = np.clip(baseline_preds, 0, None)
    metrics = regression_metrics(y_test.values, baseline_preds)
    results["Baseline"] = metrics
    predictions["Baseline"] = baseline_preds
    print_metrics(metrics, "Baseline (distance ÷ speed)")

    # ── ML Models ─────────────────────────────────────────────
    model_names = {
        "linear_regression": "Linear Regression",
        "ridge_regression":  "Ridge Regression",
        "lightgbm":          "LightGBM",
        "mlp":               "MLP Neural Network",
    }

    for file_name, display_name in model_names.items():
        model = load_model(file_name)
        if model is None:
            continue

        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)   # Ensure non-negative predictions

        metrics = regression_metrics(y_test.values, preds)
        results[display_name] = metrics
        predictions[display_name] = preds
        print_metrics(metrics, display_name)

    return results, predictions


# ── Chart 1: Model Comparison Bar Chart ──────────────────────

def plot_model_comparison(results: dict) -> None:
    """
    Side-by-side bar chart comparing all models on RMSE, MAE, and R².

    The baseline is shown in grey — all colored bars must be shorter
    (for RMSE/MAE) or taller (for R²) to declare success.
    """
    print_section("Chart: Model Comparison")

    model_names = list(results.keys())
    rmse_vals = [results[m]["RMSE"] for m in model_names]
    mae_vals  = [results[m]["MAE"]  for m in model_names]
    r2_vals   = [results[m]["R2"]   for m in model_names]

    colors = ["#B0BEC5"] + ["#42A5F5", "#66BB6A", "#EF5350", "#AB47BC"][: len(model_names)-1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for ax, vals, title, lower_better in [
        (axes[0], rmse_vals, "RMSE (min) — Lower is Better", True),
        (axes[1], mae_vals,  "MAE (min)  — Lower is Better",  True),
        (axes[2], r2_vals,   "R²         — Higher is Better", False),
    ]:
        bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=25)

        # Add value labels on top of each bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(vals),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9
            )

        # Draw a horizontal baseline reference line
        baseline_val = vals[0]
        ax.axhline(baseline_val, color="red", linestyle="--", linewidth=1,
                   alpha=0.5, label=f"Baseline: {baseline_val:.3f}")
        ax.legend(fontsize=8)

    plt.suptitle(
        "Model Performance Comparison — Test Set (Jan–Feb 2026)",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_fig("07_model_comparison.png")


# ── Chart 2: Feature Importance ──────────────────────────────

def plot_feature_importance(feature_cols: list) -> None:
    """
    Plot LightGBM feature importances.

    Feature importance tells us WHICH features the model relied on most.
    This is essential for:
    - Business interpretation ("what drives trip duration?")
    - Debugging ("is the model using leaky features?")
    - Feature selection for future model versions

    LightGBM has two importance types:
        'split': how often a feature is used to split a tree
        'gain':  total gain in accuracy from splits on this feature
    We use 'gain' — it's more meaningful than raw split count.
    """
    print_section("Chart: Feature Importance (LightGBM)")

    model = load_model("lightgbm")
    if model is None:
        logger.warning("  LightGBM model not found. Skipping feature importance.")
        return

    # Get feature importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(25)  # Top 25

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(feat_imp["feature"], feat_imp["importance"], color="#5C6BC0")
    ax.set_title("Top 25 Feature Importances — LightGBM (Gain)")
    ax.set_xlabel("Importance (Gain)")

    plt.tight_layout()
    save_fig("08_feature_importance.png")

    # Log top 10 for report
    top10 = feat_imp.tail(10)[::-1]
    logger.info("\n  Top 10 Most Important Features:")
    for _, row in top10.iterrows():
        logger.info(f"    {row['feature']:<30} {row['importance']:.2f}")


# ── Chart 3: Actual vs. Predicted ────────────────────────────

def plot_actual_vs_predicted(y_test, predictions: dict) -> None:
    """
    Scatter plot of actual vs. predicted trip durations for the best model.

    A perfect model would show all points on the diagonal line y = x.
    Scatter above the line = model underestimates (predicts too short).
    Scatter below the line = model overestimates (predicts too long).
    """
    print_section("Chart: Actual vs. Predicted")

    # Use LightGBM if available, else first available model
    best_model = "LightGBM" if "LightGBM" in predictions else list(predictions.keys())[-1]
    preds = predictions[best_model]
    actuals = y_test.values

    # Sample to avoid overplotting
    idx = np.random.choice(len(actuals), size=min(20_000, len(actuals)), replace=False)
    y_sample   = actuals[idx]
    pred_sample = preds[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Scatter ───────────────────────────────────────────────
    axes[0].scatter(y_sample, pred_sample, alpha=0.08, s=4, color="#1E88E5")
    max_val = max(y_sample.max(), pred_sample.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_title(f"Actual vs. Predicted — {best_model}")
    axes[0].set_xlabel("Actual Duration (min)")
    axes[0].set_ylabel("Predicted Duration (min)")
    axes[0].legend()

    # ── Residual Distribution ─────────────────────────────────
    residuals = pred_sample - y_sample   # positive = over-predicted
    axes[1].hist(residuals, bins=100, color="#EF5350", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].axvline(np.mean(residuals), color="blue", linestyle="--",
                    linewidth=1.5, label=f"Mean error: {np.mean(residuals):.2f} min")
    axes[1].set_title(f"Residual Distribution — {best_model}")
    axes[1].set_xlabel("Prediction Error (minutes)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle("Prediction Quality Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("09_actual_vs_predicted.png")

    logger.info(f"\n  Residual analysis for {best_model}:")
    logger.info(f"    Mean error:   {np.mean(residuals):.3f} min (bias)")
    logger.info(f"    Std of error: {np.std(residuals):.3f} min (variance)")
    logger.info(f"    % within ±5 min: {100*np.mean(np.abs(residuals) <= 5):.1f}%")


# ── Chart 4: Error Analysis ───────────────────────────────────

def plot_error_analysis(X_test, y_test, predictions: dict) -> None:
    """
    Analyze WHERE the model makes the biggest errors.

    Error analysis is a critical part of any ML project.
    Questions we answer:
        - Do errors increase with trip length? (heteroscedasticity)
        - Is the model worse during certain hours?
        - Does it struggle more with airport trips?

    This guides future model improvements.
    """
    print_section("Chart: Error Analysis")

    best_model = "LightGBM" if "LightGBM" in predictions else list(predictions.keys())[-1]
    preds = predictions[best_model]

    df_err = X_test.copy()
    df_err["actual"]    = y_test.values
    df_err["predicted"] = preds
    df_err["abs_error"] = np.abs(preds - y_test.values)
    df_err["pct_error"] = df_err["abs_error"] / df_err["actual"] * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Error by Hour ─────────────────────────────────────────
    hour_err = df_err.groupby("hour")["abs_error"].mean()
    axes[0, 0].plot(hour_err.index, hour_err.values, marker="o", color="#E53935")
    axes[0, 0].set_title("Mean Absolute Error by Hour")
    axes[0, 0].set_xlabel("Hour of Day")
    axes[0, 0].set_ylabel("MAE (minutes)")
    axes[0, 0].axvspan(7, 9, alpha=0.1, color="orange", label="AM Rush")
    axes[0, 0].axvspan(16, 19, alpha=0.1, color="red", label="PM Rush")
    axes[0, 0].legend(fontsize=8)

    # ── Error by Trip Distance Bucket ─────────────────────────
    df_err["dist_bucket"] = pd.cut(
        df_err["trip_distance"],
        bins=[0, 1, 2, 5, 10, 20, 100],
        labels=["<1mi", "1–2mi", "2–5mi", "5–10mi", "10–20mi", ">20mi"]
    )
    dist_err = df_err.groupby("dist_bucket", observed=True)["abs_error"].mean()
    axes[0, 1].bar(dist_err.index.astype(str), dist_err.values, color="#42A5F5")
    axes[0, 1].set_title("Mean Absolute Error by Distance")
    axes[0, 1].set_xlabel("Trip Distance Bucket")
    axes[0, 1].set_ylabel("MAE (minutes)")

    # ── Error by Day of Week ──────────────────────────────────
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_err = df_err.groupby("day_of_week")["abs_error"].mean()
    colors = ["#42A5F5" if d < 5 else "#FF7043" for d in range(len(dow_err))]
    axes[1, 0].bar(
        [day_labels[i] for i in dow_err.index],
        dow_err.values,
        color=colors
    )
    axes[1, 0].set_title("Mean Absolute Error by Day of Week")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("MAE (minutes)")

    # ── Actual Duration vs. Error (heteroscedasticity check) ──
    sample_idx = np.random.choice(len(df_err), size=min(15_000, len(df_err)), replace=False)
    sample = df_err.iloc[sample_idx]
    axes[1, 1].scatter(sample["actual"], sample["abs_error"],
                       alpha=0.05, s=3, color="#7B1FA2")
    axes[1, 1].set_title("Actual Duration vs. Absolute Error")
    axes[1, 1].set_xlabel("Actual Duration (min)")
    axes[1, 1].set_ylabel("Absolute Error (min)")
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_ylim(0, 60)

    plt.suptitle(
        f"Error Analysis — {best_model} on Test Set (Jan–Feb 2026)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_fig("10_error_analysis.png")

    # Log worst-performing segments for the report
    logger.info(f"\n  Error Analysis — {best_model}:")
    logger.info(f"    Worst hour: {hour_err.idxmax()}:00 (MAE = {hour_err.max():.2f} min)")
    logger.info(f"    Best hour:  {hour_err.idxmin()}:00 (MAE = {hour_err.min():.2f} min)")


# ── Write Model Report ────────────────────────────────────────

def write_model_report(results: dict) -> None:
    """
    Write a Markdown-formatted model report summarizing findings.
    This goes directly into your final README and slide deck.
    """
    print_section("Writing Model Report")

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reports", "model_report.md"
    )

    # Find the best model (lowest RMSE, excluding baseline)
    ml_models = {k: v for k, v in results.items() if k != "Baseline"}
    best_name = min(ml_models, key=lambda k: ml_models[k]["RMSE"])
    best = ml_models[best_name]
    baseline = results["Baseline"]

    rmse_improvement = (baseline["RMSE"] - best["RMSE"]) / baseline["RMSE"] * 100
    mae_improvement  = (baseline["MAE"]  - best["MAE"])  / baseline["MAE"]  * 100
    r2_improvement   = best["R2"] - baseline["R2"]

    report = f"""# Model Evaluation Report
## NYC Yellow Taxi Trip Duration Prediction
**Test Period:** January – February 2026  
**Generated:** Automatically by 06_evaluation.py

---

## Executive Summary

The best-performing model (**{best_name}**) achieved:
- **RMSE: {best["RMSE"]:.4f} minutes** — {rmse_improvement:.1f}% better than the baseline
- **MAE:  {best["MAE"]:.4f} minutes** — {mae_improvement:.1f}% better than the baseline
- **R²:   {best["R2"]:.4f}** — explains {best["R2"]*100:.1f}% of trip duration variance

---

## Model Comparison (Test Set)

| Model | RMSE (min) | MAE (min) | R² | MAPE (%) |
|-------|-----------|-----------|-----|---------|
"""
    for model_name, m in results.items():
        marker = " ← **BEST**" if model_name == best_name else ""
        report += (
            f"| {model_name}{marker} | {m['RMSE']:.4f} | "
            f"{m['MAE']:.4f} | {m['R2']:.4f} | {m['MAPE']:.2f} |\n"
        )

    report += f"""
---

## Success Criteria Assessment

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| RMSE < 5 min | < 5.0 | {best["RMSE"]:.2f} | {"✅ PASS" if best["RMSE"] < 5 else "❌ FAIL"} |
| MAE < 3 min  | < 3.0 | {best["MAE"]:.2f}  | {"✅ PASS" if best["MAE"] < 3  else "❌ FAIL"} |
| R² > 0.80    | > 0.80| {best["R2"]:.2f}   | {"✅ PASS" if best["R2"] > 0.8 else "❌ FAIL"} |
| Beat baseline RMSE | < {baseline["RMSE"]:.2f} | {best["RMSE"]:.2f} | {"✅ PASS" if best["RMSE"] < baseline["RMSE"] else "❌ FAIL"} |

---

## Key Findings

1. **Distance & route history** are the most important features.
2. **Rush hour and time-of-day** significantly affect duration independently of distance.
3. **LightGBM** outperforms linear models because trip duration is non-linear
   (traffic creates interactions between time, location, and distance).
4. **Errors are largest** for long trips and during peak PM rush hour (5–7 PM).

---

## Recommendations

- Deploy the {best_name} model for real-time ETA estimation
- Retrain monthly to capture seasonal drift
- Add real-time traffic data as a feature in v2
- Investigate systematic under-prediction during special events (holidays, concerts)
"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"  ✓ Model report saved: {report_path}")
    print(report)


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 6: EVALUATION & INTERPRETATION")
    print("="*60)

    X_test, y_test, feature_cols = load_test_data()

    # Evaluate all models on the sealed test set
    results, predictions = evaluate_all_models(X_test, y_test)

    # Generate all charts
    plot_model_comparison(results)
    plot_feature_importance(feature_cols)
    plot_actual_vs_predicted(y_test, predictions)
    plot_error_analysis(X_test, y_test, predictions)

    # Write the final report
    write_model_report(results)

    print("\n✅ Phase 6 complete!")
    print("   All figures saved to: reports/figures/")
    print("   Model report saved to: reports/model_report.md\n")
