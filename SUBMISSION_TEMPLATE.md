# Your Submission: Writeup Template

## Your final score

Dev MAE: **272.4 s**

---

## Your approach, in one paragraph

I built a high-performance **LightGBM** regressor using a modular, production-ready pipeline. The core of the approach lies in capturing the periodicity of traffic through **Cyclical Sin/Cos Encoding** for temporal features and discretizing NYC geography using **KMeans Spatial Clustering**. By calculating the **Haversine Distance** and engineering interaction terms between distance and rush-hour flags, the model effectively learns asymmetric traffic friction, significantly outperforming linear baselines.

## What you tried that didn't work

- **XGBoost (Native)**: Encountered library linkage issues on macOS (`libomp`), which led to a pivot toward `scikit-learn`'s `HistGradientBoostingRegressor` and eventually a standalone `LightGBM` implementation for better speed/memory trade-offs.
- **Raw Hour Encoding**: Initially used raw hours (0-23), but found that the model struggled to reconcile the continuity between 11 PM and Midnight until cyclical encoding was implemented.

## Where AI tooling sped you up most

AI tooling (Antigravity/Gemini) was pivotal in **refactoring the monolithic baseline into a modular architecture** (`src/` structure). It accelerated the implementation of the vectorized Haversine formula and provided the boilerplate for the **LightGBM Early Stopping** callbacks, which prevented overfitting on the 37M+ row dataset. The tool fell short on resolving the specific macOS `libomp` binary dependency for XGBoost, requiring manual intervention to switch to more portable frameworks.

## Next experiments

- **Weather Integration**: Merging NOAA hourly precipitation data to capture the high variance of speeds during rain/snow events.
- **Graph Embeddings**: Treating NYC taxi zones as a graph network to learn latent spatial relationships that clustering might oversimplify.

## How to reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare data
python data/download_data.py

# 3. Train the champion model (produces model.pkl)
python src/train.py

# 4. Evaluate on Dev set
python grade.py
```

---

_Total time spent on this challenge: 4 hours._
