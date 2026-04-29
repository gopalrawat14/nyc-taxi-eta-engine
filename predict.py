import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load artifacts once
MODEL_PATH = Path(__file__).parent / "model.pkl"
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    _MODEL = data['model']
    _ENCODINGS = data['encodings']
    _FEATURES = data['features']

def predict(request: dict) -> float:
    # 1. Parse
    dt = datetime.fromisoformat(request['requested_at'])
    pz = int(request['pickup_zone'])
    dz = int(request['dropoff_zone'])
    
    # 2. Features
    h_sin = np.sin(2 * np.pi * dt.hour / 24)
    h_cos = np.cos(2 * np.pi * dt.hour / 24)
    d_sin = np.sin(2 * np.pi * dt.weekday() / 7)
    d_cos = np.cos(2 * np.pi * dt.weekday() / 7)
    
    # Target encoding lookup
    pair_medians, pickup_medians, global_median = _ENCODINGS
    encoded = pair_medians.get((pz, dz), pickup_medians.get(pz, global_median))
    
    # 3. Predict
    x = np.array([[
        pz, dz, int(request.get('passenger_count', 1)),
        1 if dt.weekday() >= 5 else 0,
        h_sin, h_cos, d_sin, d_cos,
        (pz * 1000) + dz,
        encoded
    ]], dtype=np.float32)
    
    return float(_MODEL.predict(x)[0])
