#!/usr/bin/env python3
"""Run train/evaluate/export/benchmark for multiple WISDM seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.export.benchmark_tflite import benchmark_tflite_model
from activity_recognition.export.export_tflite import export_keras_to_tflite
from activity_recognition.training.evaluate import evaluate_from_config
from activity_recognition.training.train import load_config, train_from_config
from activity_recognition.utils.metrics import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to run."
    )
    parser.add_argument(
        "--base-output-dir", default=None, help="Parent directory for seed runs."
    )
    parser.add_argument(
        "--runs", type=int, default=500, help="Measured TFLite benchmark runs."
    )
    parser.add_argument(
        "--warmup", type=int, default=50, help="Warmup TFLite benchmark runs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_output_dir = Path(args.base_output_dir or f"{config['output']['dir']}_seeds")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        run_dir = base_output_dir / f"seed_{seed}"
        train_from_config(args.config, seed_override=seed, output_dir_override=run_dir)
        metrics = evaluate_from_config(args.config, run_dir)

        tflite_path = run_dir / "model.tflite"
        export_metadata = export_keras_to_tflite(run_dir / "model.keras", tflite_path)
        benchmark = benchmark_tflite_model(
            tflite_path,
            _input_shape(run_dir),
            runs=args.runs,
            warmup=args.warmup,
            out_path=run_dir / "benchmark.json",
        )
        training_summary = json.loads(
            (run_dir / "training_summary.json").read_text(encoding="utf-8")
        )
        results.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "model_parameters": training_summary["model_parameters"],
                "best_epoch": training_summary["best_epoch"],
                "epochs_ran": training_summary["epochs_ran"],
                "tflite_size_mb": export_metadata["tflite_size_mb"],
                "mean_ms": benchmark["mean_ms"],
                "median_ms": benchmark["median_ms"],
                "p95_ms": benchmark["p95_ms"],
            }
        )

    summary = {
        "config": str(args.config),
        "seeds": args.seeds,
        "runs": results,
        "aggregate": {
            metric: _mean_std(results, metric)
            for metric in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "mean_ms",
                "median_ms",
                "p95_ms",
            )
        },
    }
    save_json(summary, base_output_dir / "seed_summary.json")
    _write_csv(results, base_output_dir / "seed_results.csv")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


def _input_shape(run_dir: Path) -> tuple[int, int, int]:
    preprocessing = json.loads(
        (run_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    return (1, int(preprocessing["window_size"]), len(preprocessing["feature_cols"]))


def _mean_std(rows: list[dict[str, object]], metric: str) -> dict[str, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"mean": float(values.mean()), "std": std}


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
