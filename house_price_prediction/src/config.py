from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test" / "test.csv"