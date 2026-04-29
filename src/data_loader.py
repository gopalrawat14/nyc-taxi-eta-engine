import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the dataset from a CSV file."""
    try:
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values by dropping rows with critical missing info."""
    initial_rows = len(df)
    critical_columns = ['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude']
    
    # Check if columns exist before dropping
    existing_cols = [c for c in critical_columns if c in df.columns]
    df = df.dropna(subset=existing_cols)
    
    dropped = initial_rows - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to missing values.")
    
    return df

def convert_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Converts a column to datetime format."""
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
        # Drop rows where conversion failed
        df = df.dropna(subset=[column])
        logger.info(f"Converted {column} to datetime.")
    return df

def filter_outliers(df: pd.DataFrame, duration_col: str) -> pd.DataFrame:
    """Filters trip duration outliers: 10s < duration < 3 hours."""
    if duration_col not in df.columns:
        logger.error(f"Column {duration_col} not found for outlier filtering.")
        return df
    
    initial_rows = len(df)
    # 10 seconds to 10,800 seconds (3 hours)
    mask = (df[duration_col] >= 10) & (df[duration_col] <= 10800)
    df = df[mask]
    
    logger.info(f"Filtered {initial_rows - len(df)} duration outliers.")
    return df

def filter_gps_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Filters invalid GPS coordinates (New York City boundary approximation)."""
    initial_rows = len(df)
    
    # NYC Bounds: Latitude [40.5, 41.0], Longitude [-74.3, -73.6]
    coords = {
        'pickup_latitude': (40.5, 41.0),
        'pickup_longitude': (-74.3, -73.6),
        'dropoff_latitude': (40.5, 41.0),
        'dropoff_longitude': (-74.3, -73.6)
    }
    
    for col, (min_val, max_val) in coords.items():
        if col in df.columns:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
            
    logger.info(f"Filtered {initial_rows - len(df)} invalid GPS coordinate rows.")
    return df

def preprocess_pipeline(file_path: str, duration_col: str = 'trip_duration') -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_dataset(file_path)
    df = handle_missing_values(df)
    df = convert_datetime(df, 'pickup_datetime')
    df = convert_datetime(df, 'dropoff_datetime')
    df = filter_outliers(df, duration_col)
    df = filter_gps_coordinates(df)
    
    logger.info(f"Preprocessing complete. Remaining rows: {len(df)}")
    return df

if __name__ == "__main__":
    # Example usage
    # df = preprocess_pipeline('data/raw/taxi_data.csv')
    pass
