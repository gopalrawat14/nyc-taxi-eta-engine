# 🚖 NYC Taxi ETA Prediction Engine

*A high-performance, production-grade machine learning pipeline to predict trip durations with precision and scalability.*

---

## 🚀 Overview
This repository contains a modular, end-to-end machine learning system designed to solve the NYC Taxi ETA challenge. It transforms raw ride-hailing data into accurate, sub-millisecond predictions, achieving a **~24% performance boost** over standard linear baselines.

## 🎯 Problem Statement
Predicting Trip Duration (seconds) is a non-linear problem where geographical distance is only the "floor." The real complexity lies in capturing temporal traffic cycles and spatial congestion bottlenecks. The goal: Minimize **Mean Absolute Error (MAE)** while maintaining production constraints (inference speed and model size).

## 🧠 Approach
The project follows a "Baseline-First" philosophy:
1. **Robust Ingestion**: Aggressive outlier removal and GPS validation to ensure high data quality.
2. **Feature Density**: Prioritizing signal over noise by engineering interactions between space and time.
3. **Gradient Boosting**: Leveraging **LightGBM** for its superior handling of large-scale tabular data and native categorical feature support.

## ⚙️ Features Engineered
- **Haversine Distance**: Great-circle distance between pickup and dropoff coordinates.
- **Temporal Cycles**: Hour-of-day, day-of-week, and specific **Rush Hour** boolean flags.
- **Spatial Clustering**: KMeans-based neighborhood mapping to capture regional traffic friction.
- **Cross-Interactions**: Distance weighted by the hour of day (capturing varying speeds).
- **Cluster Interaction**: Encoding specific "Route IDs" based on pickup-dropoff cluster pairs.

## 🤖 Models Used
- **Baseline**: Linear Regression (provides a performance lower bound).
- **Champion**: LightGBM Regressor (GBDT) with optimized hyperparameters for tabular variance.
  - *Key Config*: Early stopping, MAE objective, and feature fractioning for robustness.

## 📊 Results
| Model | Dev MAE | % Improvement |
| :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 385.4s | - |
| **LightGBM (Champion)** | **291.2s** | **+24.4%** |

## 🔍 Key Insights
- **Spatial Bottlenecks**: Route-based cluster interactions significantly reduced error in "bridge-crossing" trips.
- **Rush Hour Sensitivity**: Temporal features are the strongest weights after distance, validating the impact of commute cycles.
- **Log Transformation**: Applying a log transform to the target helped stabilize model convergence for extreme long-distance trips.

## 🏗 Project Structure
```text
├── data/               # Raw and processed datasets
├── models/             # Versioned model artifacts (model.pkl)
├── notebooks/          # Exploratory Data Analysis (EDA)
├── outputs/            # Evaluation reports and performance plots
├── src/                # Core application logic
│   ├── config.py       # Global paths and hyperparameters
│   ├── data_loader.py  # Ingestion and cleaning pipeline
│   ├── features.py     # Advanced feature engineering
│   ├── train.py        # Model training and comparison
│   └── evaluate.py     # Error analysis and stakeholder reporting
└── requirements.txt    # Production dependencies
```

## ▶️ How to Run
### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
```bash
# This loads data, engineers features, trains the model, and runs evaluation
python -m src.train
```

### 3. Explore Insights
Open `notebooks/eda.ipynb` or check `outputs/performance_overview.png` for detailed error analysis.

## 🔮 Future Improvements
1. **Weather Integration**: Join NOAA weather data to account for speed drops during rain/snow.
2. **Deep Learning**: Experiment with TabNet or DeepFM for higher-order feature interactions.
3. **Graph Embeddings**: Represent NYC taxi zones as a graph to learn better latent spatial relationships.

---
**Author**: Applied AI Engineer | Gobblecube Challenge
