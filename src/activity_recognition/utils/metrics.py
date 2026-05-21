"""Metric helpers for activity recognition experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, object], str, np.ndarray]:
    """Compute headline metrics, text report, and confusion matrix."""

    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": report_dict,
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, report_text, cm


def top_confusions(
    confusion: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Return the most frequent off-diagonal confusion pairs."""

    pairs: list[dict[str, object]] = []
    for true_idx, true_name in enumerate(class_names):
        true_support = int(confusion[true_idx].sum())
        if true_support == 0:
            continue
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count <= 0:
                continue
            pairs.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": count,
                    "true_support": true_support,
                    "pct_of_true_class": float(count / true_support),
                }
            )

    pairs.sort(
        key=lambda item: (item["count"], item["pct_of_true_class"]), reverse=True
    )
    return pairs[:top_k]


def save_json(data: dict[str, object], path: str | Path) -> None:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
