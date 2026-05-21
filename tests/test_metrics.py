import numpy as np

from activity_recognition.utils.metrics import top_confusions


def test_top_confusions_sorts_off_diagonal_pairs():
    confusion = np.asarray(
        [
            [8, 2, 0],
            [4, 5, 1],
            [0, 3, 7],
        ]
    )

    pairs = top_confusions(confusion, ["walk", "upstairs", "downstairs"], top_k=2)

    assert pairs == [
        {
            "true_class": "upstairs",
            "predicted_class": "walk",
            "count": 4,
            "true_support": 10,
            "pct_of_true_class": 0.4,
        },
        {
            "true_class": "downstairs",
            "predicted_class": "upstairs",
            "count": 3,
            "true_support": 10,
            "pct_of_true_class": 0.3,
        },
    ]
