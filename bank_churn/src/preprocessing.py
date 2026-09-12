import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bank_churn.src.data import load_data, split_data


# Mengikuti daftar numerik pada notebook asli.
NUMERIC_FEATURES = [
    "dependent_count",
    "months_on_book",
    "total_relationship_count",
    "months_inactive_12_mon",
    "contacts_count_12_mon",
    "credit_limit",
    "total_revolving_bal",
    "total_amt_chng_q4_q1",
    "total_trans_ct",
    "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
]

PASSTHROUGH_FEATURES = [
    "gender",
    "card_category",
    "education_level",
    "Divorced",
    "Married",
    "Single",
]


def prepare_features(X):
    """Mapping dan pemilihan fitur mengikuti notebook asli."""

    X = X.copy()

    mappings = {
        "gender": {
            "M": 0,
            "F": 1,
        },
        "card_category": {
            "Blue": 0,
            "Silver": 1,
            "Gold": 2,
            "Platinum": 3,
        },
        "education_level": {
            "Uneducated": 0,
            "High School": 1,
            "College": 2,
            "Graduate": 3,
            "Post-Graduate": 4,
            "Doctorate": 5,
            "Unknown": 3,
        },
        "income_category": {
            "Less than $40K": 0,
            "$40K - $60K": 1,
            "$60K - $80K": 2,
            "$80K - $120K": 3,
            "$120K +": 4,
            "Unknown": 0,
        },
    }

    for column, mapping in mappings.items():
        X[column] = X[column].map(mapping)

        allowed_statuses = ["Divorced", "Married", "Single", "Unknown"]

    invalid_status = ~X["marital_status"].isin(allowed_statuses)

    if invalid_status.any():
        invalid_values = (
            X.loc[invalid_status, "marital_status"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"marital_status tidak valid: {invalid_values}. "
            f"Gunakan salah satu dari {allowed_statuses}."
        )

    marital_dummies = pd.get_dummies(
        X["marital_status"]
    ).reindex(
        columns=["Divorced", "Married", "Single"],
        fill_value=False,
    )

    X = pd.concat([X, marital_dummies], axis=1)

    X = X.drop(
        columns=[
            "marital_status",
            "customer_age",
            "income_category",
            "avg_open_to_buy",
            "total_trans_amt",
        ]
    )

    return X


def build_preprocessor():
    """Scaling numerik untuk Logistic Regression."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", "passthrough", PASSTHROUGH_FEATURES),
        ]
    )


def main():
    df = load_data()
    X_train, X_val, _, _, _, _ = split_data(df)

    # Data hasil mapping untuk model berbasis tree.
    X_train_ready = prepare_features(X_train)
    X_val_ready = prepare_features(X_val)

    # Periksa kesesuaian kolom tanpa mengubah hasil encoding.
    if list(X_train_ready.columns) != list(X_val_ready.columns):
        raise ValueError(
            "Kolom train dan validation berbeda setelah get_dummies. "
            "Perlu memeriksa kategori sebelum melanjutkan."
        )

    # Data hasil scaling untuk Logistic Regression.
    preprocessor = build_preprocessor()

    X_train_scaled = preprocessor.fit_transform(X_train_ready)
    X_val_scaled = preprocessor.transform(X_val_ready)

    print("Preprocessing berhasil!")
    print(f"Train mentah: {X_train.shape}")
    print(f"Train setelah mapping/drop: {X_train_ready.shape}")
    print(f"Validation setelah mapping/drop: {X_val_ready.shape}")
    print(f"Train setelah scaling: {X_train_scaled.shape}")
    print(f"Validation setelah scaling: {X_val_scaled.shape}")

    print("\nFitur yang digunakan:")
    for column in X_train_ready.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()