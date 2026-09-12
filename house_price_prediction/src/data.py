import pandas as pd

from house_price_prediction.src.config import TRAIN_PATH, TEST_PATH


def load_data():
    train = pd.read_csv(TRAIN_PATH).set_index("Id")
    test = pd.read_csv(TEST_PATH).set_index("Id")

    y_raw = train["SalePrice"].copy()

    train_feat = train.drop(columns=["SalePrice"])
    train_feat["__is_train"] = 1
    test["__is_train"] = 0

    combined = pd.concat(
        [train_feat, test],
        axis=0,
        sort=False,
    )

    train_mask = combined["__is_train"] == 1

    return train, test, y_raw, combined, train_mask


def main():
    train, test, y_raw, combined, train_mask = load_data()

    print("Dataset berhasil dibaca.")
    print(f"Train: {train.shape}")
    print(f"Test setelah penambahan penanda: {test.shape}")
    print(f"Combined: {combined.shape}")
    print(f"Baris training: {int(train_mask.sum())}")
    print(f"Baris test: {int((~train_mask).sum())}")
    print(f"Target: {y_raw.name}")


if __name__ == "__main__":
    main()