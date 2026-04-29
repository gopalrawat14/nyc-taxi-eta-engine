import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
import logging
from src.features import engineer_features
from src.config import PROCESSED_DATA_PATH, MODEL_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def compute_encodings(df: pd.DataFrame):
    logger.info("Computing target encodings...")
    pair_medians = df.groupby(['pickup_zone', 'dropoff_zone'])['duration_seconds'].median().to_dict()
    pickup_medians = df.groupby('pickup_zone')['duration_seconds'].median().to_dict()
    global_median = float(df['duration_seconds'].median())
    return pair_medians, pickup_medians, global_median

def main():
    # 1. Load
    train = pd.read_parquet(PROCESSED_DATA_PATH / "sample_1M.parquet")
    dev = pd.read_parquet(PROCESSED_DATA_PATH / "dev.parquet")
    
    # 2. Encodings
    encodings = compute_encodings(train)
    
    # 3. Features
    X_train = engineer_features(train, encodings)
    y_train = train['duration_seconds']
    
    X_dev = engineer_features(dev, encodings)
    y_dev = dev['duration_seconds']
    
    features = [
        'pickup_zone', 'dropoff_zone', 'passenger_count', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'route_id', 'encoded_duration'
    ]
    
    # 4. Train
    logger.info("Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=10,
        learning_rate=0.08,
        random_state=42
    )
    
    model.fit(X_train[features], y_train)
    
    # 5. Evaluate
    preds = model.predict(X_dev[features])
    mae = float(np.mean(np.abs(preds - y_dev)))
    logger.info(f"Dev MAE: {mae:.1f}s")
    
    # 6. Save
    logger.info(f"Saving to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'encodings': encodings, 'features': features}, f)

if __name__ == "__main__":
    main()
