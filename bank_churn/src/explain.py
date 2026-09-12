import joblib
import pandas as pd
import matplotlib

# Simpan grafik langsung ke file.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from sklearn.inspection import (
    permutation_importance,
    PartialDependenceDisplay,
)

from src.config import PROJECT_ROOT
from src.data import load_data, split_data
from src.preprocessing import prepare_features


def save_figure(path):
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()
    X_train, _, X_test, _, _, y_test = split_data(df)

    artifact = joblib.load(
        PROJECT_ROOT
        / "models"
        / "notebook_method"
        / "xgboost.joblib"
    )

    model = artifact["model"]
    expected_columns = artifact["prepared_feature_columns"]

    X_train_ready = prepare_features(X_train)
    X_test_ready = prepare_features(X_test)

    for name, frame in [
        ("Train", X_train_ready),
        ("Test", X_test_ready),
    ]:
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"Kolom {name} berbeda dari kolom saat training."
            )

    X_train_float = X_train_ready.astype(float)
    X_test_float = X_test_ready.astype(float)

    reports_dir = PROJECT_ROOT / "reports" / "notebook_method"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature importance bawaan XGBoost.
    print("1/4 Membuat feature importance...", flush=True)

    importance_df = pd.DataFrame(
        {
            "feature": X_train_float.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(
        reports_dir / "xgboost_feature_importance.csv",
        index=False,
    )

    plot_data = importance_df.sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_data["feature"],
        plot_data["importance"],
        color="steelblue",
    )
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()

    save_figure(figures_dir / "xgboost_feature_importance.png")

    # 2. SHAP menggunakan training, seperti notebook.
    print("2/4 Menghitung SHAP pada training...", flush=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_float)

    shap.summary_plot(
        shap_values,
        X_train_float,
        show=False,
    )

    save_figure(figures_dir / "xgboost_shap_summary.png")

    # 3. Permutation importance menggunakan test.
    print("3/4 Menghitung permutation importance...", flush=True)

    permutation = permutation_importance(
        model,
        X_test_float,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    permutation_df = pd.DataFrame(
        {
            "feature": X_test_float.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    permutation_df.to_csv(
        reports_dir / "xgboost_permutation_importance.csv",
        index=False,
    )

    plot_data = permutation_df.sort_values("importance_mean")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
        color="coral",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Permutation Importance — Test")
    ax.set_xlabel("Penurunan ROC-AUC setelah fitur diacak")
    fig.tight_layout()

    save_figure(figures_dir / "xgboost_permutation_importance.png")

    # 4. PDP enam fitur teratas menggunakan training.
    print("4/4 Membuat PDP...", flush=True)

    top_features = importance_df.head(6)["feature"].tolist()

    fig, ax = plt.subplots(figsize=(14, 10))

    PartialDependenceDisplay.from_estimator(
        model,
        X_train_float,
        features=top_features,
        kind="average",
        grid_resolution=50,
        ax=ax,
    )

    fig.suptitle("Partial Dependence — Top 6 Features")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_figure(figures_dir / "xgboost_pdp.png")

    print("\nTop 6 feature importance:")
    print(importance_df.head(6).to_string(index=False))

    print("\nTop 6 permutation importance:")
    print(permutation_df.head(6).to_string(index=False))

    print(f"\nGrafik tersimpan di: {figures_dir}")


if __name__ == "__main__":
    main()