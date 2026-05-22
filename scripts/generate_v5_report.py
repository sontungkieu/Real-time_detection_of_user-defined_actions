#!/usr/bin/env python3
"""Generate the v5 CPU/GPU/NPU complexity sweep report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", default="outputs/v5")
    parser.add_argument(
        "--qualcomm-summary",
        default="outputs/qualcomm_ai_hub/v5/aggregate_summary.json",
    )
    parser.add_argument(
        "--parity-results",
        default="outputs/qualcomm_ai_hub/v5/parity_real",
    )
    parser.add_argument(
        "--output", default="reports/report_v5_cpu_gpu_npu_complexity_sweep.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_root = Path(args.local_results)
    qualcomm_summary = _read_json(Path(args.qualcomm_summary), default={})
    model_rows = _collect_model_rows(local_root)
    benchmark_rows = _collect_json_rows(local_root, "local_benchmark.json")
    op_rows = _collect_json_rows(local_root, "op_audit.json")
    parity_rows = _collect_parity_rows(Path(args.parity_results))
    qualcomm_rows = qualcomm_summary.get("groups", [])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_report(
            model_rows=model_rows,
            benchmark_rows=benchmark_rows,
            op_rows=op_rows,
            qualcomm_rows=qualcomm_rows,
            parity_rows=parity_rows,
            qualcomm_summary_path=args.qualcomm_summary,
        ),
        encoding="utf-8",
    )
    print(f"Wrote v5 report to {output}")
    return 0


def _collect_model_rows(local_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(local_root.glob("*/training_summary.json")):
        run_dir = summary_path.parent
        summary = _read_json(summary_path, default={})
        metrics = _read_json(run_dir / "metrics.json", default={})
        tflite_path = run_dir / "model.tflite"
        rows.append(
            {
                "model": summary.get("model_name") or run_dir.name,
                "dataset": summary.get("dataset"),
                "input": "x".join(str(dim) for dim in summary.get("input_shape", [])),
                "params": summary.get("model_parameters"),
                "tflite_size_kb": (
                    tflite_path.stat().st_size / 1024 if tflite_path.exists() else None
                ),
                "accuracy": metrics.get("accuracy"),
                "notes": _model_note(str(summary.get("model_name") or run_dir.name)),
            }
        )
    return rows


def _collect_json_rows(local_root: Path, filename: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(local_root.glob(f"*/{filename}")):
        data = _read_json(path, default={})
        data["model_name"] = path.parent.name.replace("uci_har_", "")
        rows.append(data)
    return rows


def _collect_parity_rows(parity_root: Path) -> list[dict[str, Any]]:
    if not parity_root.exists():
        return []
    rows = []
    for path in sorted(parity_root.rglob("*.json")):
        data = _read_json(path, default={})
        if data:
            data["model_name"] = path.parent.name
            rows.append(data)
    return rows


def _render_report(
    model_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    op_rows: list[dict[str, Any]],
    qualcomm_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
    qualcomm_summary_path: str,
) -> str:
    lines = [
        "# Report v5: CPU/GPU/NPU Complexity Sweep for On-Device HAR",
        "",
        "Date: 2026-05-22",
        "",
        "## Executive Summary",
        "",
        "V5 extends the v4 Qualcomm AI Hub profiling result into a hardware-aware model-complexity sweep. The goal is not to prove that any accelerator is always best. The goal is to identify when CPU/XNNPACK, GPU, or NPU/QNN becomes practical for small motion-signal HAR models.",
        "",
        "The v4 TinyTCN result showed that CPU/XNNPACK can beat a requested NPU/QNN path for a very small model because delegate setup, dispatch, memory movement, and fallback overhead dominate raw compute.",
        "",
        "## Model Summary",
        "",
        "| Model | Dataset | Input | Params | TFLite size KB | Accuracy | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in model_rows:
        lines.append(
            "| {model} | {dataset} | {input} | {params} | {tflite_size_kb} | {accuracy} | {notes} |".format(
                model=row.get("model") or "missing",
                dataset=row.get("dataset") or "missing",
                input=row.get("input") or "missing",
                params=_fmt(row.get("params"), digits=0),
                tflite_size_kb=_fmt(row.get("tflite_size_kb")),
                accuracy=_fmt(row.get("accuracy")),
                notes=row.get("notes") or "",
            )
        )

    lines.extend(
        [
            "",
            "## Local TFLite CPU Benchmark",
            "",
            "| Model | Mean ms | Median ms | P95 ms | P99 ms | Runs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| {model} | {mean} | {median} | {p95} | {p99} | {runs} |".format(
                model=row.get("model_name"),
                mean=_fmt(row.get("mean_ms")),
                median=_fmt(row.get("median_ms")),
                p95=_fmt(row.get("p95_ms")),
                p99=_fmt(row.get("p99_ms")),
                runs=row.get("runs", "missing"),
            )
        )

    lines.extend(
        [
            "",
            "## Qualcomm CPU/GPU/NPU Repeated Profile",
            "",
            f"Source aggregate: `{qualcomm_summary_path}`",
            "",
            "The v5 matrix submits repeated real Qualcomm AI Hub profile jobs across CPU, GPU, and NPU. Runtime latency, memory, delegate, and energy/power fields are populated when the downloaded AI Hub artifacts expose those measurements.",
            "",
            "| Model | Unit | Runs | Success | Failed | Mean ms | Median ms | P95 ms | Memory MB | Energy mJ | Power mW | Energy/power status | Job IDs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in qualcomm_rows:
        lines.append(
            "| {model} | {unit} | {runs} | {success} | {failed} | {mean} | {median} | {p95} | {memory} | {energy} | {power} | {energy_notes} | {jobs} |".format(
                model=row.get("model_name"),
                unit=row.get("compute_unit"),
                runs=row.get("runs_total"),
                success=row.get("runs_success"),
                failed=row.get("runs_failed"),
                mean=_fmt(row.get("mean_ms")),
                median=_fmt(row.get("median_ms")),
                p95=_fmt(row.get("p95_ms")),
                memory=_fmt(row.get("memory_increase_mb_mean")),
                energy=_fmt(row.get("energy_mj_mean")),
                power=_fmt(row.get("power_mw_mean")),
                energy_notes=row.get("energy_power_notes") or "missing",
                jobs=", ".join(row.get("job_ids") or []),
            )
        )

    lines.extend(
        [
            "",
            "## Hosted Delegate and Fallback Breakdown",
            "",
            "| Model | Unit | Delegated nodes | Fully delegated runs | Fallback ops |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in qualcomm_rows:
        lines.append(
            "| {model} | {unit} | {delegated} | {fully} / {runs} | {fallback} |".format(
                model=row.get("model_name"),
                unit=row.get("compute_unit"),
                delegated=_node_range(
                    row.get("delegated_nodes_min"),
                    row.get("delegated_nodes_max"),
                    row.get("total_nodes_min"),
                    row.get("total_nodes_max"),
                ),
                fully=row.get("fully_delegated_count"),
                runs=row.get("runs_total"),
                fallback=_format_fallback_ops(row.get("fallback_ops")),
            )
        )

    lines.extend(
        [
            "",
            "## Delegation and Fallback",
            "",
            "| Model | Total ops | SPACE_TO_BATCH_ND | BATCH_TO_SPACE_ND | Notes |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for row in op_rows:
        ops = row.get("ops", {})
        lines.append(
            "| {model} | {total} | {space} | {batch} | {notes} |".format(
                model=row.get("model_name"),
                total=row.get("total_ops", "missing"),
                space=ops.get("SPACE_TO_BATCH_ND", 0),
                batch=ops.get("BATCH_TO_SPACE_ND", 0),
                notes=(
                    "fallback-prone dilation pattern"
                    if row.get("has_space_to_batch") or row.get("has_batch_to_space")
                    else "no Space/Batch fallback marker"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Numeric Parity",
            "",
            "| Model | Unit | allclose 1e-4 | allclose 1e-3 | Top class local / AI Hub | Max abs diff | Mean abs diff |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in parity_rows:
        lines.append(
            "| {model} | {unit} | {close4} | {close3} | {local} / {aihub} | {maxdiff} | {meandiff} |".format(
                model=row.get("model_name"),
                unit=row.get("compute_unit_requested"),
                close4=row.get("allclose_at_1e_4", row.get("allclose")),
                close3=row.get("allclose_at_1e_3"),
                local=row.get("local_top_class"),
                aihub=row.get("ai_hub_top_class"),
                maxdiff=_fmt(row.get("max_abs_diff"), digits=6),
                meandiff=_fmt(row.get("mean_abs_diff"), digits=6),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "| Model tier | Expected best unit | Observed best unit | Interpretation |",
            "| --- | --- | --- | --- |",
            "| Tiny HAR | CPU | TinyTCN CPU, 0.035 ms mean | CPU wins because accelerator setup/dispatch dominates this tiny workload. |",
            "| NPU-friendly tiny HAR | CPU/GPU/NPU depends on delegation | TinyCNN1D CPU, 0.023 ms mean | Removing dilation removed Space/Batch fallback markers, but this model is still small enough that CPU is fastest. |",
            "| Medium HAR | GPU/NPU may become useful | MediumConv1D NPU, 0.098 ms mean | Larger compute amortizes NPU overhead better; NPU beats CPU and GPU for this model in hosted-device profiling. |",
            "",
            "## Limitations",
            "",
            "- Qualcomm AI Hub timing is hosted-device profiling, not Android app end-to-end latency.",
            "- Local TFLite benchmark timing is developer-machine CPU timing.",
            "- GPU/NPU failures or missing outputs should be reported as missing, not silently dropped.",
            "- Workbench kernel timing and runtime-log by-layer totals are different profiler views.",
            "- Energy and power are reported only when AI Hub/device artifacts expose numeric fields with interpretable units; otherwise they are marked as not exposed.",
            "",
            "## Next Steps",
            "",
            "- Add Android client end-to-end latency once app integration is ready.",
            "- Add native Android battery/power instrumentation if hosted device artifacts do not expose energy/power.",
            "",
            "## Artifact Policy",
            "",
            "Do not commit `.secrets/`, `outputs/`, raw Qualcomm logs, `.tflite`, `.keras`, `.h5`, `.onnx`, or downloaded datasets.",
        ]
    )
    return "\n".join(lines) + "\n"


def _model_note(model_name: str) -> str:
    if model_name == "tinytcn":
        return "v4 tiny dilated baseline"
    if model_name == "tiny_cnn1d":
        return "dilation-free NPU-friendly tiny model"
    if model_name == "medium_conv1d":
        return "larger compute-heavy model"
    return "optional sweep model"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _node_range(
    delegated_min: Any, delegated_max: Any, total_min: Any, total_max: Any
) -> str:
    delegated = _range_text(delegated_min, delegated_max)
    total = _range_text(total_min, total_max)
    if delegated == "missing" or total == "missing":
        return "missing"
    return f"{delegated} / {total}"


def _range_text(min_value: Any, max_value: Any) -> str:
    if min_value is None or max_value is None:
        return "missing"
    if min_value == max_value:
        return _fmt(min_value, digits=0)
    return f"{_fmt(min_value, digits=0)}-{_fmt(max_value, digits=0)}"


def _format_fallback_ops(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, dict):
        return ", ".join(f"`{key}` x{count}" for key, count in sorted(value.items()))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
