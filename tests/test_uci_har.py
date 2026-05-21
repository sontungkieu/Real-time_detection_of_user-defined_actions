import numpy as np

from activity_recognition.data.uci_har import load_uci_har


def test_load_uci_har_inertial_signals(tmp_path):
    dataset_dir = tmp_path / "UCI HAR Dataset"
    for split in ("train", "test"):
        signal_dir = dataset_dir / split / "Inertial Signals"
        signal_dir.mkdir(parents=True)
        n_rows = 2 if split == "train" else 1
        for channel in ("total_acc_x", "total_acc_y", "body_gyro_x"):
            values = np.arange(n_rows * 128, dtype=np.float32).reshape(n_rows, 128)
            np.savetxt(signal_dir / f"{channel}_{split}.txt", values)
        np.savetxt(dataset_dir / split / f"y_{split}.txt", np.ones(n_rows), fmt="%d")
        np.savetxt(
            dataset_dir / split / f"subject_{split}.txt",
            np.arange(1, n_rows + 1),
            fmt="%d",
        )

    (dataset_dir / "activity_labels.txt").write_text("1 WALKING\n", encoding="utf-8")

    windows = load_uci_har(
        tmp_path,
        channels=["total_acc_x", "total_acc_y", "body_gyro_x"],
    )

    assert windows.X.shape == (3, 128, 3)
    assert windows.feature_cols == ["total_acc_x", "total_acc_y", "body_gyro_x"]
    assert windows.labels.tolist() == ["WALKING", "WALKING", "WALKING"]
    assert windows.splits.tolist() == ["train", "train", "test"]
