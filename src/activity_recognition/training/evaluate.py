"""Evaluation pipeline for saved WISDM runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from activity_recognition.data.windowing import Standardizer, transform_windows
from activity_recognition.training.train import _load_windows, load_config
from activity_recognition.utils.metrics import (
    compute_metrics,
    save_json,
    top_confusions,
)
from activity_recognition.utils.plotting import save_confusion_matrix


def evaluate_from_config(
    config_path: str | Path, run_dir: str | Path
) -> dict[str, object]:
    """Evaluate a saved model using the same config and subject split."""

    config = load_config(config_path)
    run_dir = Path(run_dir)
    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained Keras model: {model_path}")

    class_names = json.loads(
        (run_dir / "label_classes.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (run_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    split_subjects = json.loads(
        (run_dir / "split_subjects.json").read_text(encoding="utf-8")
    )

    windows = _load_windows(config)
    test_idx = np.flatnonzero(np.isin(windows.subjects, split_subjects["test"]))
    if len(test_idx) == 0:
        raise ValueError("No test windows were found for the saved subject split.")

    standardizer = Standardizer(mean=preprocessing["mean"], std=preprocessing["std"])
    X_test = transform_windows(windows.X[test_idx], standardizer)
    y_test = _encode_labels(windows.labels[test_idx], class_names)

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(X_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    metrics, report_text, cm = compute_metrics(y_test, y_pred, class_names)
    confusion_pairs = top_confusions(cm, class_names)
    metrics.update(
        {
            "num_test_windows": int(len(test_idx)),
            "test_subjects": split_subjects["test"],
            "model_path": str(model_path),
            "top_confusions": confusion_pairs,
        }
    )

    save_json(metrics, run_dir / "metrics.json")
    save_json(
        {"labels": class_names, "matrix": cm.tolist()},
        run_dir / "confusion_matrix.json",
    )
    save_json({"top_confusions": confusion_pairs}, run_dir / "confusion_analysis.json")
    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (run_dir / "confusion_analysis.txt").write_text(
        _format_confusion_analysis(confusion_pairs), encoding="utf-8"
    )
    save_confusion_matrix(cm, class_names, run_dir / "confusion_matrix.png")
    print(f"Saved metrics to {run_dir / 'metrics.json'}")
    return metrics


def _encode_labels(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    unknown = sorted({str(label) for label in labels if str(label) not in class_to_id})
    if unknown:
        raise ValueError(
            f"Found labels not present in saved label_classes.json: {unknown}"
        )
    return np.asarray([class_to_id[str(label)] for label in labels], dtype=np.int64)


def _format_confusion_analysis(confusion_pairs: list[dict[str, object]]) -> str:
    if not confusion_pairs:
        return "No off-diagonal confusions were observed.\n"
    lines = ["Top off-diagonal confusion pairs:"]
    for pair in confusion_pairs:
        lines.append(
            "- true={true_class} predicted={predicted_class}: {count}/{true_support} "
            "({pct_of_true_class:.1%})".format(**pair)
        )
    return "\n".join(lines) + "\n"
