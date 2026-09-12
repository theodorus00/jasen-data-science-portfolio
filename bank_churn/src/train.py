import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from bank_churn.src.config import PROJECT_ROOT
from bank_churn.src.data import load_data, split_data
from bank_churn.src.preprocessing import prepare_features, build_preprocessor


def main():
    df = load_data()
    X_train, X_val, _, y_train, y_val, _ = split_data(df)

    X_train_ready = prepare_features(X_train)
    X_val_ready = prepare_features(X_val)

    if list(X_train_ready.columns) != list(X_val_ready.columns):
        raise ValueError(
            "Kolom train dan validation berbeda setelah preprocessing."
        )

    # Scaling hanya untuk Logistic Regression.
    preprocessor = build_preprocessor()

    train_scaled = preprocessor.fit_transform(X_train_ready)
    val_scaled = preprocessor.transform(X_val_ready)

    scaled_columns = preprocessor.get_feature_names_out()

    X_train_scaled = pd.DataFrame(
        train_scaled,
        columns=scaled_columns,
        index=X_train_ready.index,
    )

    X_val_scaled = pd.DataFrame(
        val_scaled,
        columns=scaled_columns,
        index=X_val_ready.index,
    )

    # Konversi khusus XGBoost, mengikuti notebook.
    X_train_float = X_train_ready.astype(float)
    X_val_float = X_val_ready.astype(float)

    # Parameter model mengikuti notebook asli.
    experiments = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=1000),
            X_train_scaled,
            X_val_scaled,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
            X_train_ready,
            X_val_ready,
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=42),
            X_train_ready,
            X_val_ready,
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            X_train_float,
            X_val_float,
        ),
    ]

    # Folder terpisah agar tidak menimpa model versi sebelumnya.
    models_dir = PROJECT_ROOT / "models" / "notebook_method"
    reports_dir = PROJECT_ROOT / "reports" / "notebook_method"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for name, model, train_input, val_input in experiments:
        print(f"\n{'=' * 50}")
        print(f"Training: {name}")

        model.fit(train_input, y_train)

        # Mengikuti notebook: menggunakan predict() bawaan model.
        predictions = model.predict(val_input)

        report = classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Tidak Churn", "Churn"],
            output_dict=True,
            zero_division=0,
        )

        print(
            classification_report(
                y_val,
                predictions,
                labels=[0, 1],
                target_names=["Tidak Churn", "Churn"],
                digits=4,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")

        summary.append(
            {
                "model": name,
                "accuracy": report["accuracy"],
                "precision_churn": report["Churn"]["precision"],
                "recall_churn": report["Churn"]["recall"],
                "f1_churn": report["Churn"]["f1-score"],
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
            }
        )

        # Simpan model dan informasi preprocessing yang dibutuhkan.
        artifact = {
            "model": model,
            "model_name": name,
            "preprocessor": (
                preprocessor
                if name == "logistic_regression"
                else None
            ),
            "raw_feature_columns": X_train.columns.tolist(),
            "prepared_feature_columns": X_train_ready.columns.tolist(),
            "model_feature_columns": train_input.columns.tolist(),
            "requires_float": name == "xgboost",
            "target_mapping": {
                "Existing Customer": 0,
                "Attrited Customer": 1,
            },
        }

        joblib.dump(artifact, models_dir / f"{name}.joblib")

        pd.DataFrame(report).T.to_csv(
            reports_dir / f"{name}_classification_report.csv"
        )

        pd.DataFrame(
            matrix,
            index=["Actual Tidak Churn", "Actual Churn"],
            columns=["Predicted Tidak Churn", "Predicted Churn"],
        ).to_csv(
            reports_dir / f"{name}_confusion_matrix.csv"
        )

    results = pd.DataFrame(summary)

    results.to_csv(
        reports_dir / "model_comparison_validation.csv",
        index=False,
    )

    print("\nPERBANDINGAN MODEL PADA VALIDATION")
    print(results.round(4).to_string(index=False))

    print(f"\nModel tersimpan di: {models_dir}")
    print(f"Laporan tersimpan di: {reports_dir}")


if __name__ == "__main__":
    main()