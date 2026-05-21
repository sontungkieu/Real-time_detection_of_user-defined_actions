import numpy as np
import pandas as pd

from activity_recognition.data.splits import subject_wise_split
from activity_recognition.data.windowing import create_sliding_windows, fit_standardizer, transform_windows


def test_windowing_and_subject_split_do_not_overlap():
    rows = []
    for subject in range(4):
        for t in range(40):
            rows.append(
                {
                    "subject_id": f"s{subject}",
                    "activity": "walk",
                    "timestamp": t,
                    "x": float(t),
                    "y": float(t + 1),
                    "z": float(t + 2),
                }
            )
    df = pd.DataFrame(rows)
    windows = create_sliding_windows(
        df,
        window_size=16,
        step_size=8,
        label_col="activity",
        subject_col="subject_id",
        feature_cols=["x", "y", "z"],
        add_magnitude=True,
    )
    split = subject_wise_split(windows.subjects, 0.5, 0.25, 0.25, seed=1)
    standardizer = fit_standardizer(windows.X[split.train_idx])
    transformed = transform_windows(windows.X, standardizer)

    assert transformed.shape[-1] == 4
    assert np.isfinite(transformed).all()
    assert not set(split.train_subjects) & set(split.val_subjects)
    assert not set(split.train_subjects) & set(split.test_subjects)
