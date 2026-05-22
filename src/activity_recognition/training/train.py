"""Training pipeline for HAR experiments."""

from __future__ import annotations

import json
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder

from activity_recognition.data.splits import SubjectSplit
from activity_recognition.data.splits import subject_wise_split
from activity_recognition.data.uci_har import load_uci_har
from activity_recognition.data.windowing import (
    WindowedData,
    create_sliding_windows,
    fit_standardizer,
    transform_windows,
)
from activity_recognition.data.wisdm import load_wisdm
from activity_recognition.models.registry import build_model


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML experiment config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_from_config(
    config_path: str | Path,
    seed_override: int | None = None,
    output_dir_override: str | Path | None = None,
    model_override: str | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    learning_rate_override: float | None = None,
) -> Path:
    """Train a HAR model and return the output directory."""

    config_path = Path(config_path)
    config = _apply_runtime_overrides(
        load_config(config_path),
        seed_override,
        output_dir_override,
        model_override,
        epochs_override,
        batch_size_override,
        learning_rate_override,
    )
    _set_global_seed(int(config["split"].get("seed", 42)))

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.keras"
    if best_model_path.exists():
        best_model_path.unlink()

    windows = _load_windows(config)
    split = _make_split(windows, config)

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

    model = _build_model(
        config, input_shape=X_train.shape[1:], num_classes=len(class_names)
    )
    callbacks, callback_summary = _build_callbacks(output_dir, config, validation_data)
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=int(config["model"]["epochs"]),
        batch_size=int(config["model"]["batch_size"]),
        callbacks=callbacks,
        verbose=int(config["model"].get("verbose", 2)),
    )

    model_path = output_dir / "model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)
    model.save(model_path)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _save_json(class_names, output_dir / "label_classes.json")
    _save_json(
        {
            "feature_cols": windows.feature_cols,
            "mean": standardizer.mean,
            "std": standardizer.std,
            "window_size": int(
                config.get("window", {}).get("size", windows.X.shape[1])
            ),
            "step_size": config.get("window", {}).get("step"),
            "add_magnitude": bool(config.get("window", {}).get("add_magnitude", False)),
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
            "dataset": str(config["dataset"]["name"]),
            "split_method": str(config.get("split", {}).get("method", "subject_wise")),
            "model_path": str(model_path),
            "best_model_path": (
                str(best_model_path) if best_model_path.exists() else None
            ),
            "model_name": str(config["model"]["type"]).lower(),
            "model_parameters": int(model.count_params()),
            "input_shape": list(X_train.shape[1:]),
            "num_classes": int(len(class_names)),
            "seed": int(config["split"].get("seed", 42)),
            "commit_hash": _git_commit_hash(),
            "epochs_requested": int(config["model"]["epochs"]),
            "epochs_ran": int(len(history_frame)),
            **callback_summary,
            **_best_epoch_summary(history_frame, callback_summary),
        },
        output_dir / "training_summary.json",
    )

    print(f"Saved trained model to {model_path}")
    return output_dir


def _apply_runtime_overrides(
    config: dict[str, Any],
    seed_override: int | None,
    output_dir_override: str | Path | None,
    model_override: str | None,
    epochs_override: int | None,
    batch_size_override: int | None,
    learning_rate_override: float | None,
) -> dict[str, Any]:
    effective_config = deepcopy(config)
    if seed_override is not None:
        effective_config.setdefault("split", {})["seed"] = int(seed_override)
    if output_dir_override is not None:
        effective_config.setdefault("output", {})["dir"] = str(output_dir_override)
    if model_override is not None:
        effective_config.setdefault("model", {})["type"] = str(model_override)
    if epochs_override is not None:
        effective_config.setdefault("model", {})["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        effective_config.setdefault("model", {})["batch_size"] = int(
            batch_size_override
        )
    if learning_rate_override is not None:
        effective_config.setdefault("model", {})["learning_rate"] = float(
            learning_rate_override
        )
    return effective_config


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _load_windows(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    dataset_name = str(dataset_cfg.get("name", "")).lower()

    if dataset_name == "uci_har":
        return load_uci_har(
            dataset_cfg["raw_dir"],
            channels=dataset_cfg.get("channels"),
        )

    if dataset_name != "wisdm":
        raise ValueError(f"Unsupported dataset.name: {dataset_name}")

    window_cfg = config["window"]
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


def _make_split(windows: WindowedData, config: dict[str, Any]) -> SubjectSplit:
    split_cfg = config["split"]
    split_method = str(split_cfg.get("method", "subject_wise"))
    if split_method == "official_train_test":
        return _official_train_test_split(
            windows,
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            seed=int(split_cfg.get("seed", 42)),
        )

    if split_method != "subject_wise":
        raise ValueError(f"Unsupported split.method: {split_method}")

    return subject_wise_split(
        windows.subjects,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg.get("seed", 42)),
    )


def _official_train_test_split(
    windows: WindowedData,
    val_ratio: float,
    seed: int,
) -> SubjectSplit:
    if windows.splits is None:
        raise ValueError("official_train_test split requires dataset-provided splits.")

    split_names = windows.splits.astype(str)
    train_pool_idx = np.flatnonzero(split_names == "train")
    test_idx = np.flatnonzero(split_names == "test")
    if len(train_pool_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "official_train_test split requires non-empty train and test sets."
        )

    train_subjects_pool = np.unique(windows.subjects[train_pool_idx].astype(str))
    rng = np.random.default_rng(seed)
    shuffled_subjects = train_subjects_pool.copy()
    rng.shuffle(shuffled_subjects)

    if val_ratio > 0 and len(shuffled_subjects) > 1:
        n_val = max(1, int(round(len(shuffled_subjects) * val_ratio)))
        n_val = min(n_val, len(shuffled_subjects) - 1)
    else:
        n_val = 0

    val_subjects = shuffled_subjects[:n_val].tolist()
    train_subjects = shuffled_subjects[n_val:].tolist()

    train_idx = train_pool_idx[
        np.isin(windows.subjects[train_pool_idx], train_subjects)
    ]
    val_idx = train_pool_idx[np.isin(windows.subjects[train_pool_idx], val_subjects)]
    test_subjects = np.unique(windows.subjects[test_idx].astype(str)).tolist()

    return SubjectSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )


def _build_model(
    config: dict[str, Any], input_shape: tuple[int, int], num_classes: int
) -> tf.keras.Model:
    model_cfg = config["model"]
    model_type = model_cfg["type"].lower()
    return build_model(
        model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        config=model_cfg,
    )


def _build_callbacks(
    output_dir: Path,
    config: dict[str, Any],
    validation_data: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[list[tf.keras.callbacks.Callback], dict[str, object]]:
    callback_cfg = config.get("callbacks", {})
    early_cfg = _normalize_callback_config(callback_cfg.get("early_stopping", {}))
    checkpoint_cfg = _normalize_callback_config(callback_cfg.get("checkpoint", {}))
    monitor = str(
        early_cfg.get("monitor")
        or checkpoint_cfg.get("monitor")
        or callback_cfg.get("monitor")
        or "val_loss"
    )
    mode = str(
        early_cfg.get("mode")
        or checkpoint_cfg.get("mode")
        or _infer_monitor_mode(monitor)
    )

    summary: dict[str, object] = {
        "selection_monitor": monitor,
        "selection_mode": mode,
        "selected_model": "final_epoch",
        "early_stopping_enabled": False,
        "checkpoint_enabled": False,
    }
    if validation_data is None and monitor.startswith("val_"):
        return [], summary

    callbacks: list[tf.keras.callbacks.Callback] = []
    if bool(checkpoint_cfg.get("enabled", True)):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / "best_model.keras"),
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_weights_only=False,
                verbose=0,
            )
        )
        summary["checkpoint_enabled"] = True
        summary["selected_model"] = "best_validation_checkpoint"

    if bool(early_cfg.get("enabled", True)):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=int(early_cfg.get("patience", 5)),
                min_delta=float(early_cfg.get("min_delta", 0.0)),
                restore_best_weights=True,
                verbose=0,
            )
        )
        summary["early_stopping_enabled"] = True
        summary["early_stopping_patience"] = int(early_cfg.get("patience", 5))
        summary["early_stopping_min_delta"] = float(early_cfg.get("min_delta", 0.0))
        if not summary["checkpoint_enabled"]:
            summary["selected_model"] = "best_validation_restored_weights"

    return callbacks, summary


def _normalize_callback_config(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"enabled": value}
    if isinstance(value, dict):
        return value
    return {}


def _infer_monitor_mode(monitor: str) -> str:
    return "min" if "loss" in monitor.lower() else "max"


def _best_epoch_summary(
    history_frame: pd.DataFrame,
    callback_summary: dict[str, object],
) -> dict[str, object]:
    monitor = str(callback_summary.get("selection_monitor", "val_loss"))
    if monitor not in history_frame:
        return {"best_epoch": None, "best_metric": None, "best_metric_value": None}

    mode = str(callback_summary.get("selection_mode", _infer_monitor_mode(monitor)))
    best_idx = (
        history_frame[monitor].idxmin()
        if mode == "min"
        else history_frame[monitor].idxmax()
    )
    row = history_frame.loc[best_idx]
    summary: dict[str, object] = {
        "best_epoch": int(best_idx) + 1,
        "best_metric": monitor,
        "best_metric_value": float(row[monitor]),
    }
    for metric_name in ("loss", "accuracy", "val_loss", "val_accuracy"):
        if metric_name in row:
            summary[f"best_epoch_{metric_name}"] = float(row[metric_name])
    return summary


def _save_json(data: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
