from pathlib import Path

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
MODEL_PATH = BASE_DIR / "model.pkl"

# --- Feature Configuration ---
CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month"]
NUMERICAL_FEATURES = ["passenger_count", "is_weekend", "encoded_duration"]
TARGET = "duration_seconds"
