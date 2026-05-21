"""Loader for the original self-collected CSV format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_custom_csv(
    path_or_dir: str | Path,
    label: str | None = None,
    subject_id: str = "self",
) -> pd.DataFrame:
    """Load self-collected accelerometer CSV files.

    The original prototype writes CSVs with ``x``, ``y``, ``z``, and ``time``.
    If an ``activity`` column is missing, the label can be passed explicitly or
    inferred from filenames that look like ``timestamp-label.csv``.
    """

    paths = _collect_csv_paths(path_or_dir)
    if not paths:
        raise FileNotFoundError(f"No CSV files found at {path_or_dir}.")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = _normalize_custom_frame(frame, path, label, subject_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _collect_csv_paths(path_or_dir: str | Path) -> list[Path]:
    path = Path(path_or_dir)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return []


def _normalize_custom_frame(
    df: pd.DataFrame,
    source: Path,
    label: str | None,
    subject_id: str,
) -> pd.DataFrame:
    rename_map = {"time": "timestamp"}
    df = df.rename(columns=rename_map)
    required = {"x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    if "activity" not in df.columns:
        df["activity"] = label or _infer_label_from_filename(source) or "user_defined"
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_id

    df = df[["subject_id", "activity", "timestamp", "x", "y", "z"]].copy()
    df["source_file"] = str(source)
    for col in ("timestamp", "x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "x", "y", "z"])


def _infer_label_from_filename(path: Path) -> str | None:
    stem = path.stem
    if "-" not in stem:
        return None
    return stem.split("-", 1)[1].strip() or None
