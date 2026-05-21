"""Subject-wise splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def subject_wise_split(
    subject_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> SubjectSplit:
    """Split window indices by subject, ensuring no subject overlap."""

    subject_ids = np.asarray(subject_ids).astype(str)
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("Subject-wise evaluation requires at least two unique subjects.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}.")

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    n_subjects = len(shuffled)
    n_train = max(1, int(round(n_subjects * train_ratio)))
    n_val = int(round(n_subjects * val_ratio))

    if n_subjects >= 3 and n_val == 0 and val_ratio > 0:
        n_val = 1
    if n_train + n_val >= n_subjects:
        n_train = max(1, n_subjects - n_val - 1)
    n_test = n_subjects - n_train - n_val
    if n_subjects >= 3 and n_test == 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    train_subjects = shuffled[:n_train].tolist()
    val_subjects = shuffled[n_train : n_train + n_val].tolist()
    test_subjects = shuffled[n_train + n_val :].tolist()

    _assert_no_overlap(train_subjects, val_subjects, test_subjects)

    return SubjectSplit(
        train_idx=np.flatnonzero(np.isin(subject_ids, train_subjects)),
        val_idx=np.flatnonzero(np.isin(subject_ids, val_subjects)),
        test_idx=np.flatnonzero(np.isin(subject_ids, test_subjects)),
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )


def _assert_no_overlap(*groups: list[str]) -> None:
    seen: set[str] = set()
    for group in groups:
        current = set(group)
        overlap = seen & current
        if overlap:
            raise AssertionError(f"Subjects overlap across splits: {sorted(overlap)}")
        seen.update(current)
