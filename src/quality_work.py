import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

CLASS_NAMES = ["low/moderate proneness (0)", "elevated or higher proneness (1)"]


def evaluate(model, x_train, y_train, x_test, y_test, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "majority_class_baseline_accuracy": round(float(accuracy_score(y_test, baseline.predict(x_test))), 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["0", "1"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (test set, original class distribution)")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def tune_k(build_fn, x_train, y_train, output_dir: str | Path, ks=(1, 3, 5, 7, 9, 11, 15, 21)) -> pd.DataFrame:
    """Cross-validate k on the training set only (SMOTE is applied inside each fold)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for k in ks:
        scores = cross_val_score(build_fn(k), x_train, y_train, cv=cv, scoring="balanced_accuracy")
        rows.append({"k": k, "cv_balanced_accuracy_mean": scores.mean(), "cv_balanced_accuracy_std": scores.std()})
    table = pd.DataFrame(rows)
    table.round(4).to_csv(output_dir / "k_search.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(table["k"], table["cv_balanced_accuracy_mean"], yerr=table["cv_balanced_accuracy_std"], marker="o")
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("CV balanced accuracy")
    ax.set_title("Choosing k (5-fold CV on the training set)")
    fig.tight_layout()
    fig.savefig(output_dir / "k_search.png", dpi=150)
    plt.close(fig)
    return table
