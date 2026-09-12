import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from bank_churn.src.config import PROJECT_ROOT
from bank_churn.src.data import load_data, split_data
from bank_churn.src.preprocessing import prepare_features


def main():
    df = load_data()
    _, _, X_test, _, _, y_test = split_data(df)

    model_path = (
        PROJECT_ROOT
        / "models"
        / "notebook_method"
        / "xgboost.joblib"
    )

    artifact = joblib.load(model_path)
    model = artifact["model"]

    # Mapping dan pemilihan fitur sesuai notebook.
    X_test_ready = prepare_features(X_test)

    expected_columns = artifact["prepared_feature_columns"]

    if list(X_test_ready.columns) != expected_columns:
        raise ValueError(
            "Kolom test berbeda dari kolom training. "
            "Periksa hasil encoding sebelum melanjutkan."
        )

    # XGBoost menggunakan data float, sesuai notebook.
    X_test_float = X_test_ready.astype(float)

    # Evaluasi model tersimpan tanpa training ulang.
    predictions = model.predict(X_test_float)

    print("EVALUASI FINAL XGBOOST PADA TEST")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Tidak Churn", "Churn"],
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Tidak Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )

    reports_dir = (
        PROJECT_ROOT / "reports" / "notebook_method"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(report).T.to_csv(
        reports_dir / "xgboost_test_classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=["Actual Tidak Churn", "Actual Churn"],
        columns=["Predicted Tidak Churn", "Predicted Churn"],
    ).to_csv(
        reports_dir / "xgboost_test_confusion_matrix.csv"
    )

    print(f"\nLaporan test tersimpan di: {reports_dir}")


if __name__ == "__main__":
    main()