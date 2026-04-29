import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def encode_cyclical(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """Encodes a column into sin/cos components."""
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def engineer_features(df: pd.DataFrame, encodings: Tuple[Dict, Dict, float] = None) -> pd.DataFrame:
    """
    Transforms raw NYC Taxi request data into features.
    If encodings are provided, it applies them (Inference mode).
    If not, it's in Training mode (encodings should be computed separately).
    """
    df = df.copy()
    
    # 1. Temporal
    dt = pd.to_datetime(df['requested_at'])
    df['hour'] = dt.dt.hour
    df['dow'] = dt.dt.dayofweek
    df['month'] = dt.dt.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    # Cyclical
    df = encode_cyclical(df, 'hour', 24)
    df = encode_cyclical(df, 'dow', 7)
    
    # 2. Zone Interaction
    # The 'Route ID' is a unique integer for each pickup-dropoff pair
    df['route_id'] = (df['pickup_zone'].astype(int) * 1000) + df['dropoff_zone'].astype(int)
    
    # 3. Target Encoding (The 'Distance' proxy)
    if encodings:
        pair_medians, pickup_medians, global_median = encodings
        
        # Fast mapping
        zone_pairs = list(zip(df['pickup_zone'], df['dropoff_zone']))
        df['encoded_duration'] = pd.Series(zone_pairs).map(pair_medians)
        df['encoded_duration'] = df['encoded_duration'].fillna(df['pickup_zone'].map(pickup_medians))
        df['encoded_duration'] = df['encoded_duration'].fillna(global_median)
    
    return df
