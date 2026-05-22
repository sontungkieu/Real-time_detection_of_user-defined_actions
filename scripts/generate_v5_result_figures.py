#!/usr/bin/env python3
"""Generate v5 experiment figures from curated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MODEL_ORDER = ["tinytcn", "tiny_cnn1d", "medium_conv1d"]
MODEL_LABELS = {
    "tinytcn": "TinyTCN",
    "tiny_cnn1d": "TinyCNN1D",
    "medium_conv1d": "MediumConv1D",
}
UNIT_ORDER = ["cpu", "gpu", "npu"]
UNIT_LABELS = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU"}
UNIT_COLORS = {"cpu": "#4C78A8", "gpu": "#F58518", "npu": "#54A24B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/v5")
    parser.add_argument("--figures-dir", default="results/v5/figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    local_rows = _load_local_rows(results_dir)
    profile_rows = _load_profile_rows(results_dir)
    parity_rows = _load_parity_rows(results_dir)

    save_local_accuracy_latency(
        local_rows, figures_dir / "v5_local_accuracy_latency.png"
    )
    save_qaihub_latency(profile_rows, figures_dir / "v5_qaihub_latency_by_runtime.png")
    save_qaihub_memory_load(profile_rows, figures_dir / "v5_qaihub_memory_coldload.png")
    save_delegate_parity(
        profile_rows,
        parity_rows,
        figures_dir / "v5_delegate_and_parity.png",
    )
    print(f"Wrote v5 figures to {figures_dir}")
    return 0


def save_local_accuracy_latency(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [MODEL_LABELS[row["model"]] for row in rows]
    x_positions = list(range(len(rows)))
    accuracy = [row["accuracy"] for row in rows]
    local_mean = [row["local_mean_ms"] for row in rows]
    params_k = [row["params"] / 1000.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    axes[0].bar(labels, accuracy, color="#4C78A8", width=0.6)
    axes[0].set_title("Classification Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.86, 0.92)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[0].set_axisbelow(True)
    _annotate_values(axes[0], x_positions, accuracy, "{:.3f}", y_offset=0.004)

    sizes = [max(70.0, min(650.0, value * 2.0)) for value in params_k]
    axes[1].scatter(params_k, local_mean, s=sizes, color="#F58518", alpha=0.78)
    for row, x_value, y_value in zip(rows, params_k, local_mean, strict=True):
        axes[1].annotate(
            MODEL_LABELS[row["model"]],
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    axes[1].set_title("Local CPU TFLite Tradeoff")
    axes[1].set_xlabel("Parameters (thousands)")
    axes[1].set_ylabel("Mean latency (ms)")
    axes[1].grid(color="#DDDDDD", linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.suptitle("V5 Local Model Quality and Cost", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_qaihub_latency(rows: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    x_positions = list(range(len(MODEL_ORDER)))
    bar_width = 0.24
    offsets = {"cpu": -bar_width, "gpu": 0.0, "npu": bar_width}

    for unit in UNIT_ORDER:
        values = [_profile_value(rows, model, unit, "mean_ms") for model in MODEL_ORDER]
        p95_values = [
            _profile_value(rows, model, unit, "p95_ms") for model in MODEL_ORDER
        ]
        bars = ax.bar(
            [x + offsets[unit] for x in x_positions],
            values,
            width=bar_width,
            label=UNIT_LABELS[unit],
            color=UNIT_COLORS[unit],
        )
        for x_value, p95 in zip(
            [x + offsets[unit] for x in x_positions], p95_values, strict=True
        ):
            ax.plot([x_value - 0.07, x_value + 0.07], [p95, p95], color="#222222")
        _annotate_bars(ax, bars, "{:.3f}")

    ax.set_title("Qualcomm AI Hub Hosted-Device Latency")
    ax.set_ylabel("Mean inference latency (ms)")
    ax.set_xticks(x_positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Runtime")
    ax.text(
        0.99,
        0.97,
        "black ticks mark P95",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_qaihub_memory_load(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _grouped_bar(
        axes[0],
        rows,
        "memory_increase_mb_mean",
        "Inference Memory Increase",
        "Mean memory increase (MB)",
    )
    _grouped_bar(
        axes[1],
        rows,
        "cold_load_mean_ms",
        "Cold Load Time",
        "Mean cold load (ms)",
    )
    fig.suptitle("V5 Hosted Memory and Load Overhead", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_delegate_parity(
    profile_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    x_positions = list(range(len(MODEL_ORDER)))
    bar_width = 0.24
    offsets = {"cpu": -bar_width, "gpu": 0.0, "npu": bar_width}

    for unit in UNIT_ORDER:
        ratios = []
        for model in MODEL_ORDER:
            row = _profile_row(profile_rows, model, unit)
            ratios.append(row["delegated_nodes_min"] / row["total_nodes_min"])
        bars = axes[0].bar(
            [x + offsets[unit] for x in x_positions],
            ratios,
            width=bar_width,
            label=UNIT_LABELS[unit],
            color=UNIT_COLORS[unit],
        )
        _annotate_bars(axes[0], bars, "{:.2f}", fontsize=7)

    axes[0].set_title("Delegate Coverage")
    axes[0].set_ylabel("Delegated node ratio")
    axes[0].set_xticks(x_positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[0].set_ylim(0.0, 1.12)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].legend(title="Runtime")

    for unit in UNIT_ORDER:
        values = [
            _parity_value(parity_rows, model, unit, "max_abs_diff")
            for model in MODEL_ORDER
        ]
        bars = axes[1].bar(
            [x + offsets[unit] for x in x_positions],
            values,
            width=bar_width,
            label=UNIT_LABELS[unit],
            color=UNIT_COLORS[unit],
        )
        _annotate_bars(axes[1], bars, "{:.4f}", fontsize=7)

    axes[1].set_title("Numeric Parity Drift")
    axes[1].set_ylabel("Max absolute diff")
    axes[1].set_xticks(x_positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[1].set_axisbelow(True)
    axes[1].legend(title="Runtime")

    fig.suptitle("V5 Delegate Coverage and Numeric Parity", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_local_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODEL_ORDER:
        local_dir = results_dir / "local" / model
        metrics = _read_json(local_dir / "metrics.json")
        training = _read_json(local_dir / "training_summary.json")
        benchmark = _read_json(local_dir / "local_benchmark.json")
        rows.append(
            {
                "model": model,
                "accuracy": float(metrics["accuracy"]),
                "params": float(training["model_parameters"]),
                "local_mean_ms": float(benchmark["mean_ms"]),
            }
        )
    return rows


def _load_profile_rows(results_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(results_dir / "qualcomm" / "aggregate_summary.json")
    return list(payload["groups"])


def _load_parity_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    parity_root = results_dir / "qualcomm" / "parity"
    for model in MODEL_ORDER:
        for unit in UNIT_ORDER:
            rows.append(_read_json(parity_root / model / f"{unit}.json"))
    return rows


def _grouped_bar(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
) -> None:
    x_positions = list(range(len(MODEL_ORDER)))
    bar_width = 0.24
    offsets = {"cpu": -bar_width, "gpu": 0.0, "npu": bar_width}
    for unit in UNIT_ORDER:
        bars = ax.bar(
            [x + offsets[unit] for x in x_positions],
            [_profile_value(rows, model, unit, key) for model in MODEL_ORDER],
            width=bar_width,
            label=UNIT_LABELS[unit],
            color=UNIT_COLORS[unit],
        )
        _annotate_bars(ax, bars, "{:.3f}", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions, [MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Runtime")


def _profile_row(rows: list[dict[str, Any]], model: str, unit: str) -> dict[str, Any]:
    for row in rows:
        if row["model_name"] == model and row["compute_unit"] == unit:
            return row
    raise KeyError(f"Missing profile row for {model}/{unit}")


def _profile_value(
    rows: list[dict[str, Any]], model: str, unit: str, key: str
) -> float:
    value = _profile_row(rows, model, unit).get(key)
    if value is None:
        return 0.0
    return float(value)


def _parity_value(rows: list[dict[str, Any]], model: str, unit: str, key: str) -> float:
    for row in rows:
        if (
            _model_from_path(str(row["model"])) == model
            and row["compute_unit_requested"] == unit
        ):
            return float(row[key])
    raise KeyError(f"Missing parity row for {model}/{unit}")


def _model_from_path(path: str) -> str:
    if "tiny_cnn1d" in path:
        return "tiny_cnn1d"
    if "medium_conv1d" in path:
        return "medium_conv1d"
    if "tinytcn" in path:
        return "tinytcn"
    raise ValueError(f"Cannot infer model from path: {path}")


def _annotate_values(
    ax: plt.Axes,
    x_positions: list[int],
    values: list[float],
    fmt: str,
    y_offset: float,
) -> None:
    for x_value, value in zip(x_positions, values, strict=True):
        ax.text(
            x_value,
            value + y_offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _annotate_bars(ax: plt.Axes, bars: Any, fmt: str, fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
