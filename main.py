
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay

from cosmic_ml import train_and_evaluate, FEATURES, CLASSES


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "Dataset 41.csv"
OUT = BASE / "outputs"

OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("COSMIC IDENTITY — STELLAR OBJECT CLASSIFICATION")
    print("=" * 60)

    # Check whether the dataset exists.
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Make sure Dataset 41.csv is in the same folder as main.py."
        )

    print(f"Loading dataset: {DATA.name}")

    # Train and evaluate the machine-learning pipeline.
    model, X, y, info, m = train_and_evaluate(DATA)

    # --------------------------------------------------
    # DISPLAY DATA CLEANING REPORT
    # --------------------------------------------------
    print("\nDATA CLEANING REPORT")

    for key, value in info.items():
        print(f"{key}: {value}")

    print(f"\nTraining records: {m['train_rows']}")
    print(f"Testing records:  {m['test_rows']}")
    print(f"\nTEST ACCURACY: {m['accuracy'] * 100:.2f}%")

    # --------------------------------------------------
    # DISPLAY CLASSIFICATION REPORT
    # --------------------------------------------------
    print("\nCLASSIFICATION REPORT")

    report_df = pd.DataFrame(m["report"]).T
    print(report_df.round(3).to_string())

    # --------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------
    cm = np.asarray(m["confusion_matrix"], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Cosmic Identity — Confusion Matrix")
    fig.tight_layout()

    confusion_path = OUT / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved confusion matrix: {confusion_path.name}")

    # --------------------------------------------------
    # SAVE FEATURE IMPORTANCE CHART
    # --------------------------------------------------
    classifier = model.named_steps["classifier"]

    importance = pd.Series(
        classifier.feature_importances_,
        index=X.columns,
        name="Importance"
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))

    importance.plot(
        kind="barh",
        ax=ax,
        title="Astronomical Feature Importance"
    )

    ax.set_xlabel("Feature importance")
    fig.tight_layout()

    importance_path = OUT / "feature_importance.png"
    fig.savefig(importance_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved feature importance: {importance_path.name}")

    # --------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------
    test_results = pd.DataFrame({
        "Actual_Class": m["actual"],
        "Predicted_Class": m["predicted"],
    })

    predictions_path = OUT / "sample_predictions.csv"
    test_results.to_csv(predictions_path, index=False)

    print(f"Saved test predictions: {predictions_path.name}")

    # --------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------
    model_path = OUT / "cosmic_model.joblib"

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
        },
        model_path
    )

    print(f"Saved trained model: {model_path.name}")

    # --------------------------------------------------
    # SAVE EVALUATION METRICS
    # --------------------------------------------------
    summary = {
        "accuracy": float(m["accuracy"]),
        "training_rows": int(m["train_rows"]),
        "testing_rows": int(m["test_rows"]),
        "data_report": info,
        "classification_report": m["report"],
        "confusion_matrix": cm.tolist(),
        "labels": list(CLASSES),
        "features": list(X.columns),
    }

    metrics_path = OUT / "metrics.json"

    metrics_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8"
    )

    print(f"Saved evaluation metrics: {metrics_path.name}")

    # --------------------------------------------------
    # FINISH
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"All outputs saved in: {OUT}")
    print("\nTo launch the dashboard, run:")
    print("python -m streamlit run app.py")


if __name__ == "__main__":
    main()

