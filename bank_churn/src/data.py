import pandas as pd
from sklearn.model_selection import train_test_split

from bank_churn.src.config import (
    DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset belum ditemukan. Simpan CSV di: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"user_id", TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Kolom wajib belum tersedia: {sorted(missing_columns)}"
        )

    if df["user_id"].isna().any() or df["user_id"].duplicated().any():
        raise ValueError(
            "user_id kosong atau berulang. Periksa dataset sebelum split."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target memiliki nilai kosong.")

    unknown_labels = (
        set(df[TARGET_COLUMN].unique()) - set(TARGET_MAPPING)
    )

    if unknown_labels:
        raise ValueError(
            f"Label target tidak sesuai: {unknown_labels}. "
            "Gunakan CSV asli sebelum mapping."
        )

    return df


def split_data(df):
    # Identitas dan salinan target tidak boleh menjadi fitur.
    X = df.drop(
        columns=["user_id", TARGET_COLUMN, "churn", "age_group"],
        errors="ignore",
    )

    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)

    # Sisihkan 20% data untuk evaluasi akhir.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # 25% dari sisa 80% = 20% dari total data.
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=validation_fraction,
        stratify=y_remaining,
        random_state=RANDOM_STATE,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    df = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print(f"Dataset berhasil dibaca: {df.shape}")
    print(f"Jumlah fitur: {X_train.shape[1]}")

    for name, X_part, y_part in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        print(
            f"{name}: {len(X_part):,} baris | "
            f"Churn: {y_part.mean():.2%}"
        )


if __name__ == "__main__":
    main()