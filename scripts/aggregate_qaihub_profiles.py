#!/usr/bin/env python3
"""Aggregate Qualcomm AI Hub repeated profile summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output", default="outputs/qualcomm_ai_hub/v5/aggregate_summary.json"
    )
    parser.add_argument(
        "--markdown", default="outputs/qualcomm_ai_hub/v5/aggregate_summary.md"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    records = _load_profile_records(input_dir)
    aggregate = _aggregate(records)
    payload = {
        "input_dir": str(input_dir),
        "num_records": len(records),
        "records": records,
        "groups": aggregate,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_to_markdown(aggregate), encoding="utf-8")

    print(f"Wrote aggregate JSON to {output}")
    print(f"Wrote aggregate Markdown to {markdown}")
    return 0


def _load_profile_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "status" not in data or "compute_unit_requested" not in data:
            continue
        model_name = (
            path.parts[-3] if len(path.parts) >= 3 else Path(data["model"]).parent.name
        )
        records.append(
            {
                "summary_json": str(path),
                "model_name": data.get("model_name") or model_name,
                "model_path": data.get("model"),
                "compute_unit": data.get("compute_unit_requested"),
                "device": data.get("device"),
                "status": data.get("status"),
                "job_id": data.get("job_id"),
                "job_url": data.get("job_url"),
                "latency_ms": data.get("latency_ms"),
                "memory_mb": data.get("memory_mb"),
                "energy_mj": data.get("energy_mj"),
                "power_mw": data.get("power_mw"),
                "energy_power_available": data.get("energy_power_available"),
                "energy_power_notes": data.get("energy_power_notes"),
                "cold_load_ms": data.get("cold_load_ms"),
                "warm_load_mean_ms": data.get("warm_load_mean_ms"),
                "by_layer_ms": data.get("by_layer_ms"),
                "runtime_path": data.get("runtime_path"),
                "delegate": data.get("delegate"),
                "delegated_nodes": data.get("delegated_nodes"),
                "total_nodes": data.get("total_nodes"),
                "fully_delegated": data.get("fully_delegated"),
                "fallback_ops": data.get("fallback_ops") or {},
                "error": data.get("error"),
                "notes": data.get("notes"),
            }
        )
    return records


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("model_name") or "unknown"),
            str(record.get("compute_unit") or "unknown"),
            str(record.get("device") or "unknown"),
        )
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for (model_name, compute_unit, device), group in sorted(grouped.items()):
        latencies = _numeric_values(group, "latency_ms")
        cold_loads = _numeric_values(group, "cold_load_ms")
        memories = _numeric_values(group, "memory_mb")
        energies = _numeric_values(group, "energy_mj")
        powers = _numeric_values(group, "power_mw")
        warm_loads = _numeric_values(group, "warm_load_mean_ms")
        by_layers = _numeric_values(group, "by_layer_ms")
        delegated_nodes = _numeric_values(group, "delegated_nodes")
        total_nodes = _numeric_values(group, "total_nodes")
        fallback_ops: defaultdict[str, int] = defaultdict(int)
        for record in group:
            for op_name, count in (record.get("fallback_ops") or {}).items():
                fallback_ops[str(op_name)] += int(count)
        rows.append(
            {
                "model_name": model_name,
                "compute_unit": compute_unit,
                "device": device,
                "runs_success": sum(
                    1 for item in group if item.get("status") == "success"
                ),
                "runs_failed": sum(
                    1 for item in group if item.get("status") == "failed"
                ),
                "runs_total": len(group),
                "mean_ms": _mean(latencies),
                "std_ms": _std(latencies),
                "median_ms": _median(latencies),
                "p95_ms": _percentile(latencies, 95),
                "min_ms": min(latencies) if latencies else None,
                "max_ms": max(latencies) if latencies else None,
                "cold_load_mean_ms": _mean(cold_loads),
                "warm_load_mean_ms": _mean(warm_loads),
                "by_layer_mean_ms": _mean(by_layers),
                "memory_increase_mb_mean": _mean(memories),
                "energy_mj_mean": _mean(energies),
                "power_mw_mean": _mean(powers),
                "energy_power_available_count": sum(
                    1 for item in group if item.get("energy_power_available") is True
                ),
                "energy_power_notes": _energy_power_notes(group),
                "delegated_nodes_min": (
                    min(delegated_nodes) if delegated_nodes else None
                ),
                "delegated_nodes_max": (
                    max(delegated_nodes) if delegated_nodes else None
                ),
                "total_nodes_min": min(total_nodes) if total_nodes else None,
                "total_nodes_max": max(total_nodes) if total_nodes else None,
                "fully_delegated_count": sum(
                    1 for item in group if item.get("fully_delegated") is True
                ),
                "fallback_ops": dict(sorted(fallback_ops.items())),
                "job_ids": [item.get("job_id") for item in group if item.get("job_id")],
                "workbench_urls": [
                    item.get("job_url") for item in group if item.get("job_url")
                ],
            }
        )
    return rows


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qualcomm AI Hub Aggregate Summary",
        "",
        "| Model | Unit | Runs | Success | Failed | Mean ms | Std ms | Median ms | P95 ms | Cold load mean ms | Memory MB | Energy mJ | Power mW | Energy/power status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {compute_unit} | {runs_total} | {runs_success} | {runs_failed} | "
            "{mean_ms} | {std_ms} | {median_ms} | {p95_ms} | {cold_load_mean_ms} | "
            "{memory_increase_mb_mean} | {energy_mj_mean} | {power_mw_mean} | {energy_power_notes} |".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )
    return "\n".join(lines) + "\n"


def _energy_power_notes(group: list[dict[str, Any]]) -> str:
    if any(item.get("energy_power_available") is True for item in group):
        return "energy/power metrics exposed by at least one job artifact"
    if any(item.get("energy_power_available") is False for item in group):
        return "not exposed by downloaded AI Hub profile artifacts"
    return "not checked"


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
