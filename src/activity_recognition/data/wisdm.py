"""WISDM dataset loading utilities.

The classic WISDM activity recognition release is commonly distributed as a
semicolon-terminated text file with rows like:

    user,activity,timestamp,x,y,z;

This loader also accepts CSV files with equivalent column names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

STANDARD_COLUMNS = ["subject_id", "activity", "timestamp", "x", "y", "z"]
WISDM_COLUMN_ALIASES = {
    "user": "subject_id",
    "userid": "subject_id",
    "user_id": "subject_id",
    "subject": "subject_id",
    "subject_id": "subject_id",
    "activity": "activity",
    "label": "activity",
    "timestamp": "timestamp",
    "time": "timestamp",
    "x": "x",
    "y": "y",
    "z": "z",
}


def discover_wisdm_files(raw_dir: str | Path) -> list[Path]:
    """Return candidate WISDM raw files from a directory."""

    root = Path(raw_dir)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for pattern in ("*.txt", "*.csv", "*.data"):
        candidates.extend(root.rglob(pattern))
    return sorted(path for path in candidates if path.is_file())


def load_wisdm(raw_dir: str | Path) -> pd.DataFrame:
    """Load WISDM accelerometer-like rows into a normalized dataframe.

    Parameters
    ----------
    raw_dir:
        Directory expected to contain WISDM raw files, for example
        ``data/raw/wisdm/WISDM_ar_v1.1_raw.txt``.

    Returns
    -------
    pandas.DataFrame
        Columns include ``subject_id``, ``activity``, ``timestamp``, ``x``,
        ``y``, ``z``, and ``source_file``.
    """

    files = discover_wisdm_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No WISDM raw files found in {raw_dir}. "
            "Run scripts/download_wisdm.py or place WISDM_ar_v1.1_raw.txt there."
        )

    frames = [_load_one(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(
            f"WISDM files were found in {raw_dir}, but no valid rows were parsed."
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=STANDARD_COLUMNS)
    df["subject_id"] = df["subject_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in ("x", "y", "z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "x", "y", "z"]).reset_index(drop=True)
    return df


def _load_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".data"}:
        return _parse_wisdm_text(path)
    if suffix == ".csv":
        return _parse_wisdm_csv(path)
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _parse_wisdm_text(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed = _parse_wisdm_line(line)
            if parsed is not None:
                rows.append(parsed)
    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    if not df.empty:
        df["source_file"] = str(path)
    return df


def _parse_wisdm_line(line: str) -> list[object] | None:
    line = line.strip().rstrip(";")
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return None
    subject_id, activity, timestamp, x, y, z = parts[:6]
    try:
        return [subject_id, activity, int(timestamp), float(x), float(y), float(z)]
    except ValueError:
        return None


def _parse_wisdm_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, header=None, names=STANDARD_COLUMNS)

    normalized = _normalize_columns(df)
    if not set(STANDARD_COLUMNS).issubset(normalized.columns):
        normalized = pd.read_csv(path, header=None, names=STANDARD_COLUMNS)
    normalized = normalized[STANDARD_COLUMNS].copy()
    normalized["source_file"] = str(path)
    return normalized


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in WISDM_COLUMN_ALIASES:
            renamed[col] = WISDM_COLUMN_ALIASES[key]
    return df.rename(columns=renamed)


def describe_expected_layout() -> str:
    """Return a human-readable summary for CLI help and error messages."""

    return (
        "Expected WISDM raw files under data/raw/wisdm/. "
        "The classic file is WISDM_ar_v1.1_raw.txt with rows: "
        "user,activity,timestamp,x,y,z;"
    )


def dataframe_from_rows(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Small helper used by smoke tests and examples."""

    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)
