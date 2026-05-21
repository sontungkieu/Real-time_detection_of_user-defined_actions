#!/usr/bin/env python3
"""Generate README figures from local WISDM benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MODEL_RUNS = {
    "1D-CNN": "wisdm_cnn1d",
    "MLP": "wisdm_mlp",
}
METRICS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing local benchmark outputs.",
    )
    parser.add_argument(
        "--assets-dir",
        default="docs/assets",
        help="Directory where README figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    save_model_comparison(outputs_dir, assets_dir / "wisdm_model_comparison.png")
    save_confusion_matrix(outputs_dir, assets_dir / "wisdm_confusion_matrix.png")
    save_seed_sweep(outputs_dir, assets_dir / "wisdm_seed_sweep.png")


def save_model_comparison(outputs_dir: Path, out_path: Path) -> None:
    rows = []
    for model_name, run_name in MODEL_RUNS.items():
        metrics = _read_json(outputs_dir / run_name / "metrics.json")
        rows.append(
            {
                "model": model_name,
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
        )

    x_positions = list(range(len(rows)))
    bar_width = 0.24
    offsets = {
        "accuracy": -bar_width,
        "macro_f1": 0.0,
        "weighted_f1": bar_width,
    }
    colors = {
        "accuracy": "#4C78A8",
        "macro_f1": "#F58518",
        "weighted_f1": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for metric, label in METRICS.items():
        values = [row[metric] for row in rows]
        bars = ax.bar(
            [x + offsets[metric] for x in x_positions],
            values,
            width=bar_width,
            label=label,
            color=colors[metric],
        )
        _annotate_bars(ax, bars)

    ax.set_title("WISDM Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions, [row["model"] for row in rows])
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(outputs_dir: Path, out_path: Path) -> None:
    payload = _read_json(outputs_dir / "wisdm_cnn1d" / "confusion_matrix.json")
    labels = list(payload["labels"])
    matrix = [[int(value) for value in row] for row in payload["matrix"]]

    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("WISDM 1D-CNN Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)

    max_value = max(max(row) for row in matrix)
    threshold = max_value / 2.0
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_seed_sweep(outputs_dir: Path, out_path: Path) -> None:
    rows = _read_csv(outputs_dir / "wisdm_cnn1d_seeds" / "seed_results.csv")
    seeds = [int(row["seed"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for metric, label in METRICS.items():
        values = [float(row[metric]) for row in rows]
        ax.plot(seeds, values, marker="o", linewidth=2.0, label=label)
        for seed, value in zip(seeds, values, strict=True):
            ax.annotate(
                f"{value:.3f}",
                (seed, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    ax.set_title("WISDM 1D-CNN Seed Sweep")
    ax.set_xlabel("Subject-wise split seed")
    ax.set_ylabel("Score")
    ax.set_xticks(seeds)
    ax.set_ylim(0.55, 0.92)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
