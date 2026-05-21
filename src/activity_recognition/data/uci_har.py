"""UCI HAR inertial-signal dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from activity_recognition.data.windowing import WindowedData

DEFAULT_CHANNELS = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

ALL_INERTIAL_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def load_uci_har(
    raw_dir: str | Path,
    channels: Sequence[str] | None = None,
) -> WindowedData:
    """Load pre-windowed UCI HAR inertial signals from train/test folders."""

    dataset_dir = _resolve_dataset_dir(raw_dir)
    selected_channels = list(channels or DEFAULT_CHANNELS)
    unknown = set(selected_channels) - set(ALL_INERTIAL_CHANNELS)
    if unknown:
        raise ValueError(f"Unsupported UCI HAR channels: {sorted(unknown)}")

    activity_labels = _load_activity_labels(dataset_dir / "activity_labels.txt")
    split_arrays = []
    labels = []
    subjects = []
    splits = []

    for split_name in ("train", "test"):
        split_arrays.append(
            _load_split_signals(dataset_dir, split_name, selected_channels)
        )
        split_y = _read_int_vector(dataset_dir / split_name / f"y_{split_name}.txt")
        split_subjects = _read_int_vector(
            dataset_dir / split_name / f"subject_{split_name}.txt"
        )
        labels.extend(activity_labels[int(label)] for label in split_y)
        subjects.extend(str(subject) for subject in split_subjects)
        splits.extend([split_name] * len(split_y))

    X = np.concatenate(split_arrays, axis=0).astype(np.float32)
    if len(labels) != len(X):
        raise ValueError(
            f"Loaded {len(X)} signal windows but {len(labels)} labels from {dataset_dir}."
        )

    return WindowedData(
        X=X,
        labels=np.asarray(labels),
        subjects=np.asarray(subjects),
        feature_cols=selected_channels,
        splits=np.asarray(splits),
    )


def _resolve_dataset_dir(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    candidates = [raw_dir, raw_dir / "UCI HAR Dataset"]
    for candidate in candidates:
        if (candidate / "activity_labels.txt").exists():
            return candidate
    raise FileNotFoundError(
        "Missing UCI HAR dataset. Expected activity_labels.txt under "
        f"{raw_dir} or {raw_dir / 'UCI HAR Dataset'}."
    )


def _load_split_signals(
    dataset_dir: Path,
    split_name: str,
    channels: list[str],
) -> np.ndarray:
    signal_dir = dataset_dir / split_name / "Inertial Signals"
    arrays = []
    for channel in channels:
        path = signal_dir / f"{channel}_{split_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing UCI HAR inertial signal file: {path}")
        arrays.append(np.atleast_2d(np.loadtxt(path, dtype=np.float32)))
    return np.stack(arrays, axis=-1)


def _load_activity_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing UCI HAR activity label file: {path}")

    labels: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label_id, label_name = line.split(maxsplit=1)
        labels[int(label_id)] = label_name
    return labels


def _read_int_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing UCI HAR metadata file: {path}")
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))
