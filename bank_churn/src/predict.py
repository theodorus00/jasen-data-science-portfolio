import joblib
import pandas as pd

from bank_churn.src.config import PROJECT_ROOT
from bank_churn.src.data import load_data, split_data
from bank_churn.src.preprocessing import prepare_features


def load_artifact():
    model_path = (
        PROJECT_ROOT
        / "models"
        / "notebook_method"
        / "xgboost.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model belum ditemukan: {model_path}"
        )

    return joblib.load(model_path)


def predict_customers(customers, artifact):
    """Memprediksi satu atau beberapa customer dari data mentah."""

    if not isinstance(customers, pd.DataFrame):
        raise TypeError("Input harus berupa pandas DataFrame.")

    if customers.empty:
        raise ValueError("Data customer kosong.")

    raw_columns = artifact["raw_feature_columns"]

    missing_columns = set(raw_columns) - set(customers.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom input belum lengkap: {sorted(missing_columns)}"
        )

    # Ambil fitur dengan urutan yang sama seperti training.
    X = customers.loc[:, raw_columns].copy()

    X_ready = prepare_features(X)

    expected_columns = artifact["prepared_feature_columns"]

    if list(X_ready.columns) != expected_columns:
        raise ValueError(
            "Kolom hasil preprocessing tidak sesuai dengan model."
        )

    # Periksa hasil mapping kategori yang dipakai model.
    categorical_columns = [
        "gender",
        "education_level",
        "card_category",
    ]

    invalid_columns = [
        column
        for column in categorical_columns
        if X_ready[column].isna().any()
    ]

    if invalid_columns:
        raise ValueError(
            "Nilai kategori kosong atau tidak dikenali pada: "
            f"{invalid_columns}"
        )

    X_float = X_ready.astype(float)

    model = artifact["model"]

    # Gunakan aturan predict() yang sama seperti evaluasi.
    predictions = model.predict(X_float)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_float)[:, positive_index]

    results = pd.DataFrame(
        {
            "prediction": predictions.astype(int),
            "churn_probability": probabilities,
        },
        index=customers.index,
    )

    results["prediction_label"] = results["prediction"].map(
        {
            0: "Tidak Churn",
            1: "Churn",
        }
    )

    return results


def main():
    artifact = load_artifact()

    df = load_data()
    _, _, X_test, _, _, y_test = split_data(df)

    # Contoh satu customer dari test set.
    sample_customer = X_test.iloc[[0]].copy()

    results = predict_customers(sample_customer, artifact)

    print("CONTOH PREDIKSI SATU CUSTOMER")
    print(
        results.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    actual_label = (
        "Churn" if int(y_test.iloc[0]) == 1 else "Tidak Churn"
    )

    print(f"\nLabel aktual contoh customer: {actual_label}")
    print("Prediksi satu customer berhasil.")


if __name__ == "__main__":
    main()