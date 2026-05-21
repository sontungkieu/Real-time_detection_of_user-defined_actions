"""Sliding-window preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedData:
    X: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    feature_cols: list[str]
    splits: np.ndarray | None = None


@dataclass(frozen=True)
class Standardizer:
    mean: list[float]
    std: list[float]


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    label_col: str,
    subject_col: str,
    feature_cols: Sequence[str],
    add_magnitude: bool = False,
    timestamp_col: str = "timestamp",
) -> WindowedData:
    """Create fixed-size windows without crossing subject or label boundaries."""

    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    working = df.copy()
    features = list(feature_cols)
    missing = {label_col, subject_col, *features} - set(working.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if add_magnitude:
        working["magnitude"] = np.sqrt(
            working[features[0]] ** 2
            + working[features[1]] ** 2
            + working[features[2]] ** 2
        )
        features.append("magnitude")

    sort_cols = [subject_col, label_col]
    if timestamp_col in working.columns:
        sort_cols.append(timestamp_col)
    working = working.sort_values(sort_cols)

    windows: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []

    for (subject, label), group in working.groupby(
        [subject_col, label_col], sort=False
    ):
        values = group[features].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, step_size):
            windows.append(values[start : start + window_size])
            labels.append(str(label))
            subjects.append(str(subject))

    if not windows:
        raise ValueError(
            "No windows were created. Check window size, step size, and dataset length per subject/activity."
        )

    return WindowedData(
        X=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels),
        subjects=np.asarray(subjects),
        feature_cols=features,
    )


def fit_standardizer(X_train: np.ndarray, eps: float = 1e-6) -> Standardizer:
    """Fit channel-wise mean/std on training windows only."""

    if X_train.ndim != 3:
        raise ValueError(
            f"Expected X_train shape (windows, time, channels), got {X_train.shape}."
        )
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < eps, 1.0, std)
    return Standardizer(
        mean=mean.astype(float).tolist(), std=std.astype(float).tolist()
    )


def transform_windows(X: np.ndarray, standardizer: Standardizer) -> np.ndarray:
    """Apply a fitted channel-wise standardizer."""

    mean = np.asarray(standardizer.mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(standardizer.std, dtype=np.float32).reshape(1, 1, -1)
    return ((X.astype(np.float32) - mean) / std).astype(np.float32)
