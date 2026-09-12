"""Prepare this portfolio for Streamlit Community Cloud.
Run locally from Day 50 with the Python interpreter inside .venv312.
Checks trusted local artifacts and predictions, then exports runtime dependencies.
No model training, Git staging, uploads, or deployment are performed.
"""
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parent
RUNTIME_PACKAGES = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "xgboost": "xgboost",
}
REQUIRED = [
    "app.py",
    "bank_churn/data/raw/bank_churn.csv",
    "bank_churn/models/notebook_method/xgboost.joblib",
    "house_price_prediction/models/notebook_method/elasticnet_bundle.joblib",
    "house_price_prediction/data/train/train.csv",
    "house_price_prediction/data/test/test.csv",
    "house_price_prediction/reports/notebook_method/model_comparison.csv",
    "google_stock_price/reports/historical/comparison.txt",
    "google_stock_price/reports/historical/train_validation_split.png",
] + [
    f"google_stock_price/reports/historical/model_{i}_{kind}.png"
    for i in (1, 2, 3) for kind in ("loss", "predictions")
]


def check_files():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise ValueError("Required files are missing:\n" + "\n".join(missing))
    large = [name for name in REQUIRED if (ROOT / name).stat().st_size >= 95 * 1024 * 1024]
    if large:
        raise ValueError("Use external artifact storage or Git LFS for these large files:\n" + "\n".join(large))


def check_runtime():
    versions = {}
    for distribution, module in RUNTIME_PACKAGES.items():
        imported = importlib.import_module(module)
        version = metadata.version(distribution)
        actual = getattr(imported, "__version__", version)
        if actual != version:
            raise ValueError(f"{distribution}: imported version {actual} differs from installed metadata {version}. Repair the environment first.")
        versions[distribution] = version
        print(f"  {distribution}=={version}")
    return versions


def check_predictions():
    import numpy as np
    import pandas as pd
    from sklearn.exceptions import InconsistentVersionWarning
    from bank_churn.src.data import load_data, split_data
    from bank_churn.src.predict import load_artifact, predict_customers
    from house_price_prediction.src.config import TEST_PATH
    from house_price_prediction.src.predict_notebook import load_notebook_artifact, predict_houses
    # A model must load under the same sklearn version before exporting its runtime.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bank_model = load_artifact()
        frame = load_data()
        _, _, sample, _, _, _ = split_data(frame)
        bank = predict_customers(sample.head(3), bank_model)
        if len(bank) != 3 or not bank["churn_probability"].between(0, 1).all():
            raise ValueError("Bank churn prediction check failed.")
        state = load_notebook_artifact()
        house = predict_houses(pd.read_csv(TEST_PATH).head(3), state)
        if len(house) != 3 or not np.isfinite(house["SalePrice"]).all() or not house["SalePrice"].ge(0).all():
            raise ValueError("House prediction check failed.")
    print("PASS: bank churn and house-price models loaded and predicted successfully.")


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use the project's Python 3.12: .\\.venv312\\Scripts\\python.exe prepare_cloud.py")
    try:
        check_files()
        print("Runtime versions:")
        versions = check_runtime()
        check_predictions()
        old = ROOT / "requirements.txt"
        backup = ROOT / "requirements-training.txt"
        if old.exists() and not backup.exists():
            backup.write_bytes(old.read_bytes())
        requirements = (
            "# Streamlit inference runtime, exported from the checked Python 3.12 environment.\n"
            "# Training/development requirements are preserved in requirements-training.txt.\n"
            + "".join(f"{name}=={version}\n" for name, version in versions.items())
        )
        temporary = ROOT / "requirements-cloud.tmp"
        temporary.write_text(requirements, encoding="utf-8")
        temporary.replace(old)
    except Exception as error:
        print(f"\nPREPARATION STOPPED: {type(error).__name__}: {error}")
        print("Fix the reported issue before deploying. Send this output for help.")
        return 1
    print("\nLOCAL CHECKS PASSED. requirements.txt now contains the inference dependencies.")
    print("The old dependency list is preserved in requirements-training.txt.")
    print("Next: commit requirements and the required model/data files, then deploy app.py using Python 3.12.")
    print("Linux dependency installation and hosted behavior still require verification on Streamlit Cloud.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
