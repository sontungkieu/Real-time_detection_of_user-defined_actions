#!/usr/bin/env python3
"""Prepare curated v5 experiment results for version control."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODELS = {
    "tinytcn": "uci_har_tinytcn",
    "tiny_cnn1d": "uci_har_tiny_cnn1d",
    "medium_conv1d": "uci_har_medium_conv1d",
}
LOCAL_FILES = [
    "training_summary.json",
    "metrics.json",
    "local_benchmark.json",
    "op_audit.json",
    "confusion_matrix.json",
    "confusion_analysis.json",
    "history.csv",
    "label_classes.json",
    "model.tflite.json",
    "preprocessing.json",
    "split_subjects.json",
]
COMPUTE_UNITS = ["cpu", "gpu", "npu"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs/v5")
    parser.add_argument("--qualcomm-dir", default="outputs/qualcomm_ai_hub/v5")
    parser.add_argument("--results-dir", default="results/v5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    qualcomm_dir = Path(args.qualcomm_dir)
    results_dir = Path(args.results_dir)

    if not outputs_dir.exists():
        raise FileNotFoundError(f"Local v5 outputs not found: {outputs_dir}")
    if not qualcomm_dir.exists():
        raise FileNotFoundError(f"Qualcomm v5 outputs not found: {qualcomm_dir}")

    _prepare_local_results(outputs_dir, results_dir / "local")
    _prepare_qualcomm_results(qualcomm_dir, results_dir / "qualcomm")
    _write_manifest(results_dir)
    print(f"Prepared curated v5 results under {results_dir}")
    return 0


def _prepare_local_results(outputs_dir: Path, destination: Path) -> None:
    for model_name, run_name in LOCAL_MODELS.items():
        source_dir = outputs_dir / run_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Required local run not found: {source_dir}")
        target_dir = destination / model_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in LOCAL_FILES:
            source = source_dir / filename
            if not source.exists():
                continue
            _copy_curated(source, target_dir / filename)


def _prepare_qualcomm_results(source_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ["aggregate_summary.json", "aggregate_summary.md"]:
        _copy_curated(source_dir / filename, destination / filename)

    manifest_dir = destination / "repeated_real"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["manifest.csv", "manifest.json"]:
        _copy_curated(
            source_dir / "repeated_real" / filename,
            manifest_dir / filename,
        )

    for model_name in LOCAL_MODELS:
        for unit in COMPUTE_UNITS:
            for run_path in sorted(
                (source_dir / "repeated_real" / model_name / unit).glob("run_*.json")
            ):
                _copy_curated(
                    run_path,
                    destination / "profile_runs" / model_name / unit / run_path.name,
                )
            _copy_curated(
                source_dir / "parity_real" / model_name / f"{unit}.json",
                destination / "parity" / model_name / f"{unit}.json",
            )


def _copy_curated(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required result file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(
            json.dumps(_sanitize(data), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        shutil.copyfile(source, destination)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _sanitize_text(value: str) -> str:
    try:
        path = Path(value)
    except ValueError:
        return value
    if path.is_absolute():
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return path.name
    return value


def _write_manifest(results_dir: Path) -> None:
    manifest = {
        "name": "v5 CPU/GPU/NPU complexity sweep curated results",
        "local_models": list(LOCAL_MODELS),
        "compute_units": COMPUTE_UNITS,
        "profile_matrix": {
            "models": len(LOCAL_MODELS),
            "compute_units": len(COMPUTE_UNITS),
            "runs_per_pair": 5,
            "total_profile_jobs": len(LOCAL_MODELS) * len(COMPUTE_UNITS) * 5,
        },
        "numeric_parity_jobs": len(LOCAL_MODELS) * len(COMPUTE_UNITS),
        "excluded": [
            "raw Qualcomm runtime logs",
            "downloaded AI Hub profile artifacts",
            "TFLite files",
            "Keras model files",
            "raw datasets",
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
