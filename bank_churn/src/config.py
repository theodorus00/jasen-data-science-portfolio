from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank_churn.csv"

TARGET_COLUMN = "attrition_flag"

TARGET_MAPPING = {
    "Existing Customer": 0,
    "Attrited Customer": 1,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20