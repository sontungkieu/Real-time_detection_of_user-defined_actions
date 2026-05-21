"""Training pipeline for WISDM experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder

from activity_recognition.data.splits import subject_wise_split
from activity_recognition.data.windowing import create_sliding_windows, fit_standardizer, transform_windows
from activity_recognition.data.wisdm import load_wisdm
from activity_recognition.models.cnn1d import build_cnn1d
from activity_recognition.models.mlp import build_mlp


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML experiment config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_from_config(config_path: str | Path) -> Path:
    """Train a WISDM model and return the output directory."""

    config_path = Path(config_path)
    config = load_config(config_path)
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    windows = _load_windows(config)
    split_cfg = config["split"]
    split = subject_wise_split(
        windows.subjects,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg.get("seed", 42)),
    )

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(windows.labels)
    class_names = label_encoder.classes_.astype(str).tolist()

    standardizer = fit_standardizer(windows.X[split.train_idx])
    X_all = transform_windows(windows.X, standardizer)

    y_train = y_all[split.train_idx]
    X_train = X_all[split.train_idx]
    validation_data = None
    if len(split.val_idx) > 0:
        validation_data = (X_all[split.val_idx], y_all[split.val_idx])

    model = _build_model(config, input_shape=X_train.shape[1:], num_classes=len(class_names))
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=int(config["model"]["epochs"]),
        batch_size=int(config["model"]["batch_size"]),
        verbose=1,
    )

    model_path = output_dir / "model.keras"
    model.save(model_path)

    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    _save_json(class_names, output_dir / "label_classes.json")
    _save_json(
        {
            "feature_cols": windows.feature_cols,
            "mean": standardizer.mean,
            "std": standardizer.std,
            "window_size": int(config["window"]["size"]),
            "step_size": int(config["window"]["step"]),
            "add_magnitude": bool(config["window"].get("add_magnitude", False)),
        },
        output_dir / "preprocessing.json",
    )
    _save_json(
        {
            "train": split.train_subjects,
            "val": split.val_subjects,
            "test": split.test_subjects,
        },
        output_dir / "split_subjects.json",
    )
    _save_json(
        {
            "num_windows": int(len(windows.X)),
            "num_train_windows": int(len(split.train_idx)),
            "num_val_windows": int(len(split.val_idx)),
            "num_test_windows": int(len(split.test_idx)),
            "num_subjects": int(len(set(windows.subjects))),
            "classes": class_names,
            "model_path": str(model_path),
            "model_parameters": int(model.count_params()),
        },
        output_dir / "training_summary.json",
    )

    print(f"Saved trained model to {model_path}")
    return output_dir


def _load_windows(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    window_cfg = config["window"]
    if dataset_cfg.get("name") != "wisdm":
        raise ValueError("Only dataset.name=wisdm is supported by this training pipeline.")

    df = load_wisdm(dataset_cfg["raw_dir"])
    return create_sliding_windows(
        df,
        window_size=int(window_cfg["size"]),
        step_size=int(window_cfg["step"]),
        label_col=dataset_cfg["label_col"],
        subject_col=dataset_cfg["subject_col"],
        feature_cols=dataset_cfg["feature_cols"],
        add_magnitude=bool(window_cfg.get("add_magnitude", False)),
    )


def _build_model(config: dict[str, Any], input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    model_cfg = config["model"]
    model_type = model_cfg["type"].lower()
    learning_rate = float(model_cfg.get("learning_rate", 0.001))
    if model_type == "mlp":
        return build_mlp(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
    if model_type == "cnn1d":
        return build_cnn1d(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
    raise ValueError(f"Unsupported model.type: {model_type}")


def _save_json(data: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
