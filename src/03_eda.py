# ============================================================
# src/03_eda.py
# Phase 3: Exploratory Data Analysis (EDA)
#
# PURPOSE:
#   Understand the structure, patterns, and anomalies in the
#   cleaned 2025 trip data before building any models.
#   Every visualization is saved to reports/figures/.
#
# KEY QUESTIONS WE ANSWER:
#   - What does the distribution of trip durations look like?
#   - When (hour, day, month) are trips longest/shortest?
#   - Which pickup/dropoff zones are busiest?
#   - How strongly does distance correlate with duration?
#   - Are there anomalies or seasonal patterns?
#
# HOW TO RUN:
#   python src/03_eda.py
#
# OUTPUT:
#   reports/figures/*.png  (all plots saved automatically)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_logger, get_data_path, get_figures_path,
    print_section, extract_time_features
)

logger = get_logger("03_eda")

# ── Plot Styling ──────────────────────────────────────────────
# Set a consistent visual style for all charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":       120,
    "figure.figsize":   (12, 5),
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

FIGURES_DIR = get_figures_path()
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename: str) -> None:
    """Save the current matplotlib figure to reports/figures/."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved figure: {filename}")


# ── Load Data ─────────────────────────────────────────────────

def load_clean_data() -> pd.DataFrame:
    """
    Load the cleaned 2025 training data.
    For EDA we only use 2025 — never peek at the 2026 test set.
    """
    processed_dir = get_data_path("processed")
    path = os.path.join(processed_dir, "trips_2025_clean.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Cleaned data not found. Run 02_prepare_data.py first."
        )

    logger.info(f"  Loading: {path}")
    df = pd.read_parquet(path)
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── EDA Section 1: Duration Distribution ─────────────────────

def plot_duration_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of trip durations.

    WHY THIS MATTERS:
        Understanding the target variable's shape tells us:
        - Whether to use linear models (needs normality) or tree models
        - Whether the data has extreme outliers affecting training
        - What 'typical' trip lengths look like for business context

    WHAT TO LOOK FOR:
        - Right skew is expected (most trips short, few very long)
        - Multiple peaks may indicate different trip types
        - Any remaining outliers after cleaning
    """
    print_section("1. Trip Duration Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram
    axes[0].hist(
        df["trip_duration_minutes"],
        bins=100,
        color="#2196F3",
        edgecolor="white",
        linewidth=0.3
    )
    axes[0].set_title("Distribution of Trip Duration")
    axes[0].set_xlabel("Duration (minutes)")
    axes[0].set_ylabel("Number of Trips")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f} min"
    ))

    # Right: Box plot to show percentiles and any outliers
    axes[1].boxplot(
        df["trip_duration_minutes"],
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#90CAF9"),
        medianprops=dict(color="#1565C0", linewidth=2)
    )
    axes[1].set_title("Trip Duration — Box Plot")
    axes[1].set_xlabel("Duration (minutes)")
    axes[1].set_yticks([])

    plt.suptitle(
        "Trip Duration Distribution — NYC Yellow Taxi 2025",
        fontsize=15, fontweight="bold", y=1.02
    )

    # Print key statistics to console for the report
    stats = df["trip_duration_minutes"].describe(percentiles=[.25,.5,.75,.90,.95,.99])
    logger.info(f"\n  Duration Statistics:\n{stats.to_string()}")

    save_fig("01_duration_distribution.png")


# ── EDA Section 2: Temporal Patterns ─────────────────────────

def plot_temporal_patterns(df: pd.DataFrame) -> None:
    """
    Reveal how trip duration varies across time dimensions.

    WHY THIS MATTERS:
        Traffic in NYC is highly cyclical. Rush hour, weekends,
        and seasonal events all affect how long trips take.
        These patterns justify our time-based feature engineering.

    CHARTS:
        - Average duration by hour of day
        - Average duration by day of week
        - Average duration by month
        - Trip volume heatmap: hour × day of week
    """
    print_section("2. Temporal Patterns")

    # Add time features if not already present
    df = extract_time_features(df.copy(), "tpep_pickup_datetime")

    # ── 2a: By Hour ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    hour_avg = df.groupby("hour")["trip_duration_minutes"].mean()
    axes[0, 0].plot(hour_avg.index, hour_avg.values, marker="o", color="#E53935")
    axes[0, 0].fill_between(hour_avg.index, hour_avg.values, alpha=0.15, color="#E53935")
    axes[0, 0].set_title("Average Duration by Hour of Day")
    axes[0, 0].set_xlabel("Hour (0 = midnight)")
    axes[0, 0].set_ylabel("Avg Duration (min)")
    axes[0, 0].axvspan(7, 9, alpha=0.1, color="orange", label="AM Rush")
    axes[0, 0].axvspan(16, 19, alpha=0.1, color="red", label="PM Rush")
    axes[0, 0].legend(fontsize=9)

    # ── 2b: By Day of Week ────────────────────────────────────
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_avg = df.groupby("day_of_week")["trip_duration_minutes"].mean()
    colors = ["#42A5F5" if d < 5 else "#FF7043" for d in range(7)]
    axes[0, 1].bar(day_labels, day_avg.values, color=colors)
    axes[0, 1].set_title("Average Duration by Day of Week")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].set_ylabel("Avg Duration (min)")

    # ── 2c: By Month ──────────────────────────────────────────
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_avg = df.groupby("month")["trip_duration_minutes"].mean()
    axes[1, 0].bar(
        [month_labels[m-1] for m in month_avg.index],
        month_avg.values,
        color="#66BB6A"
    )
    axes[1, 0].set_title("Average Duration by Month (2025)")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Avg Duration (min)")
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # ── 2d: Heatmap — Volume by Hour × Day ───────────────────
    pivot = df.pivot_table(
        values="trip_duration_minutes",
        index="hour",
        columns="day_of_week",
        aggfunc="mean"
    )
    pivot.columns = day_labels
    sns.heatmap(
        pivot, ax=axes[1, 1], cmap="YlOrRd",
        cbar_kws={"label": "Avg Duration (min)"}
    )
    axes[1, 1].set_title("Avg Duration Heatmap: Hour × Day")
    axes[1, 1].set_xlabel("Day of Week")
    axes[1, 1].set_ylabel("Hour of Day")

    plt.suptitle(
        "Temporal Patterns — NYC Yellow Taxi 2025",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    save_fig("02_temporal_patterns.png")

    # Log key findings for the report
    peak_hour = hour_avg.idxmax()
    slow_hour = hour_avg.idxmin()
    logger.info(f"\n  Peak duration hour: {peak_hour}:00 ({hour_avg[peak_hour]:.1f} min avg)")
    logger.info(f"  Fastest hour:       {slow_hour}:00 ({hour_avg[slow_hour]:.1f} min avg)")


# ── EDA Section 3: Distance vs Duration ──────────────────────

def plot_distance_vs_duration(df: pd.DataFrame) -> None:
    """
    Scatter plot of trip distance vs trip duration.

    WHY THIS MATTERS:
        - High correlation validates distance as a key feature
        - Scatter in the relationship shows where other features
          (like traffic) add predictive value beyond pure distance
        - The baseline model assumes a linear relationship here

    WHAT TO LOOK FOR:
        - General positive trend (longer distance = longer duration)
        - Wide spread at any distance = traffic unpredictability
        - Clusters that may indicate different trip types
    """
    print_section("3. Distance vs Duration Relationship")

    # Sample 50,000 rows for visualization — plotting 30M rows freezes everything
    sample = df.sample(min(50_000, len(df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Scatter plot (sampled)
    axes[0].scatter(
        sample["trip_distance"],
        sample["trip_duration_minutes"],
        alpha=0.1, s=3, color="#1E88E5"
    )
    axes[0].set_title("Distance vs Duration (50K sample)")
    axes[0].set_xlabel("Trip Distance (miles)")
    axes[0].set_ylabel("Trip Duration (minutes)")
    axes[0].set_xlim(0, 25)
    axes[0].set_ylim(0, 120)

    # Right: Correlation heatmap for all numeric features
    numeric_cols = [
        "trip_duration_minutes", "trip_distance",
        "fare_amount", "total_amount", "passenger_count"
    ]
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, ax=axes[1], mask=mask,
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5
    )
    axes[1].set_title("Correlation Matrix — Numeric Features")

    plt.suptitle(
        "Feature Correlations — NYC Yellow Taxi 2025",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    save_fig("03_distance_vs_duration.png")

    corr_val = df["trip_distance"].corr(df["trip_duration_minutes"])
    logger.info(f"\n  Pearson correlation (distance vs duration): {corr_val:.4f}")


# ── EDA Section 4: Top Zones ──────────────────────────────────

def plot_top_zones(df: pd.DataFrame) -> None:
    """
    Bar charts of the top 20 pickup and dropoff zones by volume.

    WHY THIS MATTERS:
        - Identifies where the model will be used most
        - Shows zone imbalance (e.g., airports, midtown dominate)
        - Helps us decide whether to encode zones as categories
          or derive aggregate features
    """
    print_section("4. Spatial Patterns — Top Zones")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Load zone lookup to get zone names
    zone_path = os.path.join(get_data_path("raw"), "taxi_zone_lookup.csv")
    if os.path.exists(zone_path):
        zones = pd.read_csv(zone_path)[["LocationID", "Zone", "Borough"]]
        zones.columns = ["LocationID", "Zone", "Borough"]
    else:
        # Fallback: use numeric IDs if lookup not available
        zones = None

    for ax, col, title in [
        (axes[0], "PULocationID", "Top 20 Pickup Zones"),
        (axes[1], "DOLocationID", "Top 20 Dropoff Zones")
    ]:
        counts = df[col].value_counts().head(20).reset_index()
        counts.columns = ["LocationID", "count"]

        if zones is not None:
            counts = counts.merge(zones, on="LocationID", how="left")
            labels = counts["Zone"].fillna(counts["LocationID"].astype(str))
        else:
            labels = counts["LocationID"].astype(str)

        ax.barh(labels[::-1], counts["count"][::-1], color="#5C6BC0")
        ax.set_title(title)
        ax.set_xlabel("Number of Trips")
        ax.tick_params(axis="y", labelsize=8)
        # Format x-axis as thousands
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

    plt.suptitle(
        "Most Active Pickup & Dropoff Zones — NYC Yellow Taxi 2025",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    save_fig("04_top_zones.png")


# ── EDA Section 5: Duration by RatecodeID ────────────────────

def plot_by_ratecode(df: pd.DataFrame) -> None:
    """
    Compare trip duration distributions by RatecodeID.

    RatecodeID legend (TLC data dictionary):
        1 = Standard rate
        2 = JFK (flat rate)
        3 = Newark (flat rate)
        4 = Nassau or Westchester
        5 = Negotiated fare
        6 = Group ride

    WHY THIS MATTERS:
        JFK and Newark airport trips have fundamentally different
        duration patterns than standard trips. This helps justify
        including RatecodeID as a model feature.
    """
    print_section("5. Duration by Rate Code")

    rate_labels = {
        1: "Standard", 2: "JFK", 3: "Newark",
        4: "Nassau/Westchester", 5: "Negotiated", 6: "Group"
    }

    fig, ax = plt.subplots(figsize=(13, 5))

    rate_data = [
        df[df["RatecodeID"] == code]["trip_duration_minutes"].dropna().values
        for code in [1, 2, 3, 4, 5, 6]
        if (df["RatecodeID"] == code).any()
    ]
    rate_labels_used = [
        rate_labels.get(code, str(code))
        for code in [1, 2, 3, 4, 5, 6]
        if (df["RatecodeID"] == code).any()
    ]

    ax.boxplot(
        rate_data,
        labels=rate_labels_used,
        patch_artist=True,
        boxprops=dict(facecolor="#B2EBF2"),
        medianprops=dict(color="#00838F", linewidth=2),
        showfliers=False    # Hide outlier dots to keep chart readable
    )
    ax.set_title("Trip Duration by Rate Code — NYC Yellow Taxi 2025")
    ax.set_xlabel("Rate Code")
    ax.set_ylabel("Duration (minutes)")

    plt.tight_layout()
    save_fig("05_duration_by_ratecode.png")


# ── EDA Section 6: Anomaly Detection ─────────────────────────

def plot_speed_distribution(df: pd.DataFrame) -> None:
    """
    Compute and plot the distribution of implied speed
    (trip_distance / trip_duration_minutes * 60 = mph).

    WHY THIS MATTERS:
        Implied speed helps us:
        1. Catch remaining data quality issues (speed > 100 mph = invalid)
        2. Understand traffic variability across the dataset
        3. Validate our distance-speed baseline assumption
        4. Compute what the 'average NYC taxi speed' is for baseline

    FORMULA:
        speed_mph = (trip_distance / trip_duration_minutes) * 60
    """
    print_section("6. Implied Speed Distribution (Anomaly Check)")

    df = df.copy()
    df["speed_mph"] = (df["trip_distance"] / df["trip_duration_minutes"]) * 60

    # Remove physical impossibilities
    df_valid = df[(df["speed_mph"] > 0) & (df["speed_mph"] <= 80)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df_valid["speed_mph"], bins=100, color="#AB47BC", edgecolor="white", linewidth=0.3)
    ax.axvline(df_valid["speed_mph"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean speed: {df_valid['speed_mph'].mean():.1f} mph")
    ax.axvline(df_valid["speed_mph"].median(), color="blue", linestyle="--", linewidth=1.5,
               label=f"Median speed: {df_valid['speed_mph'].median():.1f} mph")
    ax.set_title("Distribution of Implied Trip Speed — NYC Yellow Taxi 2025")
    ax.set_xlabel("Speed (mph)")
    ax.set_ylabel("Number of Trips")
    ax.legend()

    plt.tight_layout()
    save_fig("06_speed_distribution.png")

    avg_speed = df_valid["speed_mph"].mean()
    logger.info(f"\n  Average NYC taxi speed (2025): {avg_speed:.2f} mph")
    logger.info(
        f"  This value is used to compute the BASELINE model:"
        f"\n    baseline_duration = (trip_distance / {avg_speed:.2f}) * 60"
    )

    # Save average speed for use in modeling script
    avg_speed_path = os.path.join(get_data_path("processed"), "avg_speed.txt")
    with open(avg_speed_path, "w") as f:
        f.write(str(avg_speed))
    logger.info(f"  Average speed saved to: {avg_speed_path}")


# ── Main EDA Runner ───────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NYC YELLOW TAXI — PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("="*60)

    df = load_clean_data()

    # Run all EDA sections
    plot_duration_distribution(df)
    plot_temporal_patterns(df)
    plot_distance_vs_duration(df)
    plot_top_zones(df)
    plot_by_ratecode(df)
    plot_speed_distribution(df)

    print(f"\n✅ Phase 3 complete. All figures saved to: reports/figures/")
    print("   Next step: python src/04_feature_engineering.py\n")
