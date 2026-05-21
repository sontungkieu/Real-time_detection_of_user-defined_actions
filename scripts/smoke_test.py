#!/usr/bin/env python3
"""Run a lightweight synthetic smoke test for the research pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.data.splits import subject_wise_split
from activity_recognition.data.windowing import create_sliding_windows, fit_standardizer, transform_windows
from activity_recognition.models.cnn1d import build_cnn1d
from activity_recognition.models.mlp import build_mlp


def main() -> None:
    df = _synthetic_motion_dataframe()
    windows = create_sliding_windows(
        df,
        window_size=32,
        step_size=16,
        label_col="activity",
        subject_col="subject_id",
        feature_cols=["x", "y", "z"],
        add_magnitude=True,
    )
    split = subject_wise_split(
        windows.subjects,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    standardizer = fit_standardizer(windows.X[split.train_idx])
    X = transform_windows(windows.X, standardizer)
    num_classes = len(set(windows.labels))

    mlp = build_mlp(X.shape[1:], num_classes=num_classes, verbose=False)
    cnn = build_cnn1d(X.shape[1:], num_classes=num_classes, verbose=False)

    sample = X[:2]
    assert mlp(sample).shape == (2, num_classes)
    assert cnn(sample).shape == (2, num_classes)
    assert not set(split.train_subjects) & set(split.test_subjects)
    print("Smoke test passed: windowing, subject split, MLP, and 1D-CNN are usable.")


def _synthetic_motion_dataframe() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    activities = ["walk", "run"]
    for subject in range(5):
        for activity_id, activity in enumerate(activities):
            for t in range(96):
                phase = t / 8.0
                scale = 1.0 + activity_id
                rows.append(
                    {
                        "subject_id": f"s{subject}",
                        "activity": activity,
                        "timestamp": t,
                        "x": np.sin(phase) * scale + rng.normal(0, 0.02),
                        "y": np.cos(phase) * scale + rng.normal(0, 0.02),
                        "z": 9.8 + activity_id * 0.2 + rng.normal(0, 0.02),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
