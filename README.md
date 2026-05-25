# NYC Yellow Taxi Trip Duration Prediction

> **Role:** Data Science Consultant
> **Training Data:** October-December 2025 (11.5M trips)
> **Test Data:** January-February 2026 (6.6M trips)
> **Best Model:** LightGBM - RMSE: 4.74 min | R2: 0.88

---

## Problem Statement

NYC yellow taxi dispatchers and passengers lack accurate
trip duration estimates, causing:
- Poor driver scheduling and fleet utilization
- Inaccurate fare estimates for passengers
- Inability to predict demand surges by zone and time

**Goal:** Predict how many minutes a yellow taxi trip
will take using only information available at pickup time.

---

## Objectives

### Business Objectives
1. Improve ETA communication to passengers
2. Identify unpredictable time windows and zones
3. Outperform naive baseline: duration = distance / speed

### Technical Objectives
1. Achieve RMSE < 5 minutes on Jan-Feb 2026 test set
2. Achieve MAE < 3 minutes
3. Achieve R-squared > 0.80
4. Beat the distance-speed baseline on all metrics

### Why NYC Yellow Taxi Data?
- Publicly available and regularly updated by NYC TLC
- Millions of real trips with precise timestamps
- Industry-standard benchmark for transportation ML

---

## Results - All Success Criteria Met

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| RMSE | < 5.0 min | 4.74 min | PASS |
| MAE | < 3.0 min | 2.29 min | PASS |
| R-squared | > 0.80 | 0.88 | PASS |
| Beat Baseline | < 13.53 | 4.74 | PASS |
| Within 5 min | - | 87.7% | BONUS |

---

## Dataset

- **Source:** NYC TLC Trip Record Data
- **URL:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Training:** October - December 2025
- **Test:** January - February 2026
- **Training rows:** 11,485,385 after cleaning
- **Test rows:** 6,633,847 after cleaning
- **Features:** 29 engineered input features

### Note on Data Size
The full 14-month dataset contains approximately 55 million rows.
Due to RAM constraints, we trained on the most recent 3 months
(Oct-Dec 2025) giving 11 million real NYC taxi trips.
The temporal gap between training (Dec 2025) and test (Jan 2026)
is minimal, ensuring fair and realistic evaluation.

---

## Models Compared

| Model | RMSE | MAE | R2 | Notes |
|-------|------|-----|----|-------|
| Baseline | 13.53 | 7.93 | 0.05 | No ML used |
| Linear Regression | 7.35 | 4.69 | 0.72 | Benchmark |
| Ridge Regression | 7.10 | 4.57 | 0.74 | alpha=1000 |
| LightGBM | 4.74 | 2.29 | 0.88 | BEST MODEL |
| XGBoost | 4.82 | 2.34 | 0.88 | Close second |
| MLP Neural Net | 22.88 | 20.38 | -1.70 | RAM limited |

---

## Key Findings

1. fare_amount and trip_distance are the strongest predictors
2. Route history (avg duration per PU-DO pair) is highly predictive
3. PM rush hour adds 40-60% to average trip duration
4. LightGBM beats linear models by 35%+ RMSE
5. 87.7% of test trips predicted within 5 minutes
6. Mean prediction bias is only 0.66 minutes

---

## How to Run

1. Install: pip install -r requirements.txt
2. Open: jupyter notebook
3. Run: NYC_Taxi_Project.ipynb all cells top to bottom
4. Dashboard: streamlit run src/streamlit_app.py
5. API: uvicorn src.api:app --reload
6. Tests: pytest tests/test_preprocessing.py -v

---

## Dependencies

- pandas 2.2.3
- numpy 2.1.3
- lightgbm 4.6.0
- xgboost 3.2.0
- scikit-learn
- matplotlib / seaborn
- streamlit / fastapi

---

## Author

**[Samuel Omoyayi]**
Data Science Student
GitHub: [https://github.com/WSamuelY/nyc-taxi-duration-prediction]
