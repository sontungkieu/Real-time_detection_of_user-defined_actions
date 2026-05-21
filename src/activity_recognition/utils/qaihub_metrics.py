"""Parse Qualcomm AI Hub profile logs into structured metrics."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

KB_PER_MB = 1024.0

STATUS_RE = re.compile(
    r"Status Successfully (?P<phase>Compiled|Loaded Cold|Loaded Warm|Performed Inference|Performed Inference By Layer) "
    r"with t = (?P<time_us>[0-9.]+) us and usage: (?P<usage>.*)$"
)
MEMORY_RE = re.compile(
    r"before = (?P<before>[0-9.]+) kB; "
    r"peakBefore = (?P<peak_before>[0-9.]+) kB; "
    r"mallocUnusedBefore = (?P<malloc_unused_before>[0-9.]+) kB; "
    r"after = (?P<after>[0-9.]+) kB; "
    r"peakAfter = (?P<peak_after>[0-9.]+) kB; "
    r"mallocUnusedAfter = (?P<malloc_unused_after>[0-9.]+) kB; "
    r"increase = (?P<increase_min>[0-9.]+)-(?P<increase_max>[0-9.]+) kB; "
    r"peak = (?P<peak_delta_min>[0-9.]+)-(?P<peak_delta_max>[0-9.]+) kB"
)
BY_LAYER_RE = re.compile(
    r"Populating InferByLayer results - node=(?P<node>[^,]+), "
    r"tag=(?P<tag>.*), time=(?P<time_us>[0-9.]+)us, cycles=(?P<cycles>[0-9.]+)\."
)
DELEGATE_RE = re.compile(
    r"Replacing (?P<delegated>\d+) out of (?P<total>\d+) node\(s\) with delegate "
    r"\((?P<delegate_type>[^)]+)\) node, yielding (?P<partitions>\d+) partitions"
)
APPLIED_RE = re.compile(
    r"Applied \d+ delegates?: (?P<delegate>[^.]+)\. Model is fully delegated=(?P<fully>true|false)"
)
QNN_RE = re.compile(
    r"Loaded QNN Delegate, API version=(?P<api>[^,]+), "
    r"QNN version=(?P<version>[^,]+), capabilities: (?P<capabilities>[^.]+)"
)
WALL_TIME_RE = re.compile(
    r"Successfully ran model for (?P<iterations>\d+) iterations across "
    r"(?P<batches>\d+) batches in (?P<seconds>[0-9.]+) sec"
)


def parse_runtime_log(path: str | Path) -> dict[str, Any]:
    """Parse a Qualcomm AI Hub runtime log."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    record: dict[str, Any] = {
        "source_log": str(source),
        "job_id": None,
        "model_name": None,
        "device": {},
        "compute_unit_observed": None,
        "delegate": {
            "name": None,
            "delegate_type": None,
            "delegated_nodes": None,
            "total_nodes": None,
            "partitions": None,
            "fully_delegated": None,
        },
        "qnn": {},
        "timings": {
            "compiled_us": None,
            "cold_load_us": None,
            "warm_load_us": [],
            "inference_us": None,
            "by_layer_us": None,
            "inference_wall_time_100_iter_sec": None,
            "by_layer_wall_time_100_iter_sec": None,
        },
        "memory": {
            "compiled": None,
            "cold_load": None,
            "warm_load": [],
            "inference": None,
            "by_layer": None,
            "unit": "MB",
            "source_unit": "kB",
        },
        "by_layer": {"entries": [], "summary": {}},
        "diagnostics": {"warning_count": 0, "error_count": 0, "unique_messages": []},
    }

    current_task = None
    unique_diagnostics: Counter[str] = Counter()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        job_match = re.search(r"\[job_id: (?P<job_id>j[a-z0-9]+)\]", line)
        if job_match:
            record["job_id"] = job_match.group("job_id")
        model_match = re.search(r"\[job_id: [^\]]+\] \[(?P<model>[^\]]+)\]", line)
        if model_match:
            record["model_name"] = model_match.group("model")

        prop_match = re.search(
            r"Android system property: (?P<key>[\w.]+) = (?P<value>.*)$", line
        )
        if prop_match:
            record["device"][prop_match.group("key")] = prop_match.group("value")

        opencl_match = re.search(r"OpenCL Version: (?P<value>.*)$", line)
        if opencl_match:
            record["device"]["opencl_version"] = opencl_match.group("value")

        litert_match = re.search(
            r"Loaded LiteRT version (?P<litert>[^ ]+) \(which is TF Lite version (?P<tflite>[^)]+)\)",
            line,
        )
        if litert_match:
            record["device"]["litert_version"] = litert_match.group("litert")
            record["device"]["tflite_version"] = litert_match.group("tflite")

        task_match = re.search(r"Tungsten running task: (?P<task>[^=]+?) -=-", line)
        if task_match:
            current_task = task_match.group("task").strip()

        compute_match = re.search(r"using compute unit=(?P<compute_unit>[a-z_]+)", line)
        if compute_match:
            record["compute_unit_observed"] = compute_match.group("compute_unit")

        delegate_match = DELEGATE_RE.search(line)
        if delegate_match:
            record["delegate"].update(
                {
                    "delegated_nodes": int(delegate_match.group("delegated")),
                    "total_nodes": int(delegate_match.group("total")),
                    "delegate_type": delegate_match.group("delegate_type"),
                    "partitions": int(delegate_match.group("partitions")),
                }
            )

        applied_match = APPLIED_RE.search(line)
        if applied_match:
            record["delegate"]["name"] = applied_match.group("delegate").strip()
            record["delegate"]["fully_delegated"] = (
                applied_match.group("fully") == "true"
            )

        qnn_match = QNN_RE.search(line)
        if qnn_match:
            record["qnn"] = {
                "api_version": qnn_match.group("api").strip(),
                "version": qnn_match.group("version").strip(),
                "capabilities": _parse_capabilities(qnn_match.group("capabilities")),
            }

        wall_match = WALL_TIME_RE.search(line)
        if wall_match:
            wall = {
                "iterations": int(wall_match.group("iterations")),
                "batches": int(wall_match.group("batches")),
                "seconds": float(wall_match.group("seconds")),
            }
            if current_task == "performing inference by layer":
                record["timings"]["by_layer_wall_time_100_iter_sec"] = wall["seconds"]
            elif current_task == "performing inference":
                record["timings"]["inference_wall_time_100_iter_sec"] = wall["seconds"]
            record.setdefault("wall_times", []).append(wall)

        status_match = STATUS_RE.search(line)
        if status_match:
            phase = status_match.group("phase")
            time_us = float(status_match.group("time_us"))
            memory = parse_memory_usage(status_match.group("usage"))
            _record_phase_metric(record, phase, time_us, memory)

        by_layer_match = BY_LAYER_RE.search(line)
        if by_layer_match:
            tag = by_layer_match.group("tag").strip()
            entry = {
                "node": by_layer_match.group("node"),
                "tag": tag,
                "time_us": float(by_layer_match.group("time_us")),
                "cycles": int(float(by_layer_match.group("cycles"))),
                "placement": infer_log_entry_placement(
                    tag, record.get("compute_unit_observed")
                ),
                "is_delegate": "delegate" in tag.lower(),
            }
            entry["is_fallback"] = is_fallback_entry(
                entry, record.get("compute_unit_observed")
            )
            record["by_layer"]["entries"].append(entry)

        if "warning" in line.lower() or "error" in line.lower():
            level = "warning" if "warning" in line.lower() else "error"
            record["diagnostics"][f"{level}_count"] += 1
            message = _strip_log_prefix(line)
            unique_diagnostics[message] += 1

    record["by_layer"]["summary"] = summarize_layer_entries(
        record["by_layer"]["entries"]
    )
    record["diagnostics"]["unique_messages"] = [
        {"message": message, "count": count}
        for message, count in unique_diagnostics.most_common(20)
    ]
    _finalize_timing_summary(record)
    return record


def parse_memory_usage(usage: str) -> dict[str, float] | None:
    """Parse a Tungsten memory usage string and convert values from kB to MB."""

    match = MEMORY_RE.search(usage)
    if not match:
        return None
    return {
        f"{key}_mb": round(float(value) / KB_PER_MB, 6)
        for key, value in match.groupdict().items()
    }


def parse_runtime_analysis_text(text: str, job_id: str | None = None) -> dict[str, Any]:
    """Parse a copied Qualcomm AI Hub Runtime Analysis table."""

    tabular = _parse_runtime_analysis_tsv(text, job_id)
    if tabular is not None:
        return tabular

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"job_id": job_id, "entries": [], "summary": summarize_layer_entries([])}

    header_tokens = {
        "Layer",
        "Type",
        "Kernel(s)",
        "Placement",
        "Compute Cycles",
        "Timing",
    }
    while lines and lines[0] in header_tokens:
        lines.pop(0)

    entries: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        if index + 3 >= len(lines):
            break
        layer = lines[index]
        op_type = lines[index + 1]
        index += 2

        kernels: list[str] = []
        while index < len(lines) and not _is_runtime_analysis_placement(lines[index]):
            kernels.append(lines[index])
            index += 1
        if index >= len(lines):
            break

        placement = lines[index]
        index += 1

        compute_cycles: int | None = None
        if index < len(lines) and not _looks_like_timing(lines[index]):
            compute_cycles = _parse_int(lines[index])
            index += 1
        if index >= len(lines):
            break

        timing_us = _parse_timing_us(lines[index])
        index += 1

        entry = {
            "layer": layer,
            "type": op_type,
            "kernels": kernels,
            "tag": ";".join(kernels) if kernels else op_type,
            "placement": placement,
            "time_us": timing_us,
            "cycles": compute_cycles,
            "is_delegate": placement.startswith("NPU"),
            "is_fallback": placement.startswith("CPU"),
        }
        entries.append(entry)

    return {
        "job_id": job_id,
        "source": "runtime_analysis",
        "entries": entries,
        "summary": summarize_layer_entries(entries),
    }


def summarize_layer_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize placement, delegate, fallback, timing, and cycles by layer entries."""

    placement_counter: Counter[str] = Counter()
    placement_time_us: defaultdict[str, float] = defaultdict(float)
    placement_cycles: defaultdict[str, int] = defaultdict(int)
    tag_counter: Counter[str] = Counter()
    tag_time_us: defaultdict[str, float] = defaultdict(float)
    fallback_ops: Counter[str] = Counter()

    delegate_entries = 0
    fallback_entries = 0
    total_time_us = 0.0
    total_cycles = 0

    for entry in entries:
        placement = entry.get("placement") or "unknown"
        tag = entry.get("tag") or entry.get("type") or "unknown"
        time_us = float(entry.get("time_us") or 0.0)
        cycles = int(entry.get("cycles") or 0)

        placement_counter[placement] += 1
        placement_time_us[placement] += time_us
        placement_cycles[placement] += cycles
        tag_counter[tag] += 1
        tag_time_us[tag] += time_us
        total_time_us += time_us
        total_cycles += cycles

        if entry.get("is_delegate"):
            delegate_entries += 1
        if entry.get("is_fallback"):
            fallback_entries += 1
            fallback_ops[tag] += 1

    return {
        "total_entries": len(entries),
        "delegate_entries": delegate_entries,
        "fallback_entries": fallback_entries,
        "total_time_us": round(total_time_us, 6),
        "total_cycles": total_cycles,
        "placement_counts": dict(placement_counter),
        "placement_time_us": {
            key: round(value, 6) for key, value in placement_time_us.items()
        },
        "placement_cycles": dict(placement_cycles),
        "fallback_ops": dict(fallback_ops),
        "top_tags_by_time_us": [
            {"tag": tag, "count": tag_counter[tag], "time_us": round(time_us, 6)}
            for tag, time_us in sorted(
                tag_time_us.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ],
    }


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    timings = record.get("timings", {})
    memory = record.get("memory", {})
    delegate = record.get("delegate", {})
    by_layer_summary = record.get("by_layer", {}).get("summary", {})
    runtime_analysis_summary = record.get("runtime_analysis", {}).get("summary", {})

    inference_memory = memory.get("inference") or {}
    by_layer_memory = memory.get("by_layer") or {}
    cold_load_memory = memory.get("cold_load") or {}

    return {
        "job_id": record.get("job_id"),
        "source_log": record.get("source_log"),
        "compute_unit_observed": record.get("compute_unit_observed"),
        "delegate_name": delegate.get("name"),
        "delegate_type": delegate.get("delegate_type"),
        "delegated_nodes": delegate.get("delegated_nodes"),
        "total_nodes": delegate.get("total_nodes"),
        "delegate_partitions": delegate.get("partitions"),
        "fully_delegated": delegate.get("fully_delegated"),
        "cold_load_ms": _us_to_ms(timings.get("cold_load_us")),
        "warm_load_min_ms": _us_to_ms(timings.get("warm_load_us_min")),
        "warm_load_mean_ms": _us_to_ms(timings.get("warm_load_us_mean")),
        "warm_load_max_ms": _us_to_ms(timings.get("warm_load_us_max")),
        "inference_ms": _us_to_ms(timings.get("inference_us")),
        "by_layer_ms": _us_to_ms(timings.get("by_layer_us")),
        "inference_wall_time_100_iter_sec": timings.get(
            "inference_wall_time_100_iter_sec"
        ),
        "by_layer_wall_time_100_iter_sec": timings.get(
            "by_layer_wall_time_100_iter_sec"
        ),
        "cold_load_increase_min_mb": cold_load_memory.get("increase_min_mb"),
        "cold_load_increase_max_mb": cold_load_memory.get("increase_max_mb"),
        "cold_load_peak_delta_min_mb": cold_load_memory.get("peak_delta_min_mb"),
        "cold_load_peak_delta_max_mb": cold_load_memory.get("peak_delta_max_mb"),
        "inference_increase_min_mb": inference_memory.get("increase_min_mb"),
        "inference_increase_max_mb": inference_memory.get("increase_max_mb"),
        "inference_peak_delta_min_mb": inference_memory.get("peak_delta_min_mb"),
        "inference_peak_delta_max_mb": inference_memory.get("peak_delta_max_mb"),
        "by_layer_increase_min_mb": by_layer_memory.get("increase_min_mb"),
        "by_layer_increase_max_mb": by_layer_memory.get("increase_max_mb"),
        "by_layer_peak_delta_min_mb": by_layer_memory.get("peak_delta_min_mb"),
        "by_layer_peak_delta_max_mb": by_layer_memory.get("peak_delta_max_mb"),
        "log_layer_entries": by_layer_summary.get("total_entries"),
        "log_fallback_entries": by_layer_summary.get("fallback_entries"),
        "runtime_analysis_entries": runtime_analysis_summary.get("total_entries"),
        "runtime_analysis_fallback_entries": runtime_analysis_summary.get(
            "fallback_entries"
        ),
        "warning_count": record.get("diagnostics", {}).get("warning_count"),
        "error_count": record.get("diagnostics", {}).get("error_count"),
    }


def layer_rows_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    job_id = record.get("job_id")
    for entry in record.get("by_layer", {}).get("entries", []):
        rows.append(
            {
                "job_id": job_id,
                "source": "runtime_log",
                "layer": None,
                "node": entry.get("node"),
                "type": None,
                "tag": entry.get("tag"),
                "placement": entry.get("placement"),
                "time_us": entry.get("time_us"),
                "cycles": entry.get("cycles"),
                "is_delegate": entry.get("is_delegate"),
                "is_fallback": entry.get("is_fallback"),
            }
        )
    for entry in record.get("runtime_analysis", {}).get("entries", []):
        rows.append(
            {
                "job_id": job_id,
                "source": "runtime_analysis",
                "layer": entry.get("layer"),
                "node": None,
                "type": entry.get("type"),
                "tag": entry.get("tag"),
                "placement": entry.get("placement"),
                "time_us": entry.get("time_us"),
                "cycles": entry.get("cycles"),
                "is_delegate": entry.get("is_delegate"),
                "is_fallback": entry.get("is_fallback"),
            }
        )
    return rows


def write_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_log_entry_placement(tag: str, compute_unit: str | None) -> str:
    tag_lower = tag.lower()
    if "qnn" in tag_lower:
        return "NPU (QNN)"
    if "xnnpack" in tag_lower:
        return "CPU (XNNPACK)"
    if compute_unit == "cpu_and_npu":
        if tag.strip().upper() in {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"}:
            return "CPU (TfLite fallback)"
        return "NPU (QNN kernel)"
    return "CPU (TfLite)"


def is_fallback_entry(entry: dict[str, Any], compute_unit: str | None) -> bool:
    if entry.get("is_delegate"):
        return False
    placement = str(entry.get("placement") or "")
    return compute_unit == "cpu_and_npu" and placement.startswith("CPU")


def _record_phase_metric(
    record: dict[str, Any],
    phase: str,
    time_us: float,
    memory: dict[str, float] | None,
) -> None:
    timings = record["timings"]
    memory_store = record["memory"]
    if phase == "Compiled":
        timings["compiled_us"] = time_us
        memory_store["compiled"] = memory
    elif phase == "Loaded Cold":
        timings["cold_load_us"] = time_us
        memory_store["cold_load"] = memory
    elif phase == "Loaded Warm":
        timings["warm_load_us"].append(time_us)
        memory_store["warm_load"].append(memory)
    elif phase == "Performed Inference":
        timings["inference_us"] = time_us
        memory_store["inference"] = memory
    elif phase == "Performed Inference By Layer":
        timings["by_layer_us"] = time_us
        memory_store["by_layer"] = memory


def _finalize_timing_summary(record: dict[str, Any]) -> None:
    warm_values = record["timings"].get("warm_load_us", [])
    if warm_values:
        record["timings"]["warm_load_us_min"] = min(warm_values)
        record["timings"]["warm_load_us_mean"] = mean(warm_values)
        record["timings"]["warm_load_us_max"] = max(warm_values)


def _parse_capabilities(value: str) -> dict[str, bool | str]:
    capabilities: dict[str, bool | str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, raw_value = [item.strip() for item in part.split("=", 1)]
        if raw_value.lower() in {"true", "false"}:
            capabilities[key] = raw_value.lower() == "true"
        else:
            capabilities[key] = raw_value
    return capabilities


def _strip_log_prefix(line: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", line).strip()


def _is_runtime_analysis_placement(value: str) -> bool:
    return value.startswith("NPU (") or value.startswith("CPU (")


def _looks_like_timing(value: str) -> bool:
    return value.endswith("μs") or value.endswith("us")


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _parse_timing_us(value: str) -> float:
    cleaned = value.replace("μs", "").replace("us", "").strip()
    return float(cleaned.replace(",", ""))


def _us_to_ms(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1000.0, 6)


def _parse_runtime_analysis_tsv(text: str, job_id: str | None) -> dict[str, Any] | None:
    if "\t" not in text:
        return None

    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if not reader.fieldnames:
        return None

    field_map = {
        _normalize_header(field): field
        for field in reader.fieldnames
        if field is not None
    }
    required = {"layer", "type", "placement", "timing"}
    if not required.issubset(field_map):
        return None

    entries: list[dict[str, Any]] = []
    kernels_field = field_map.get("kernel_s") or field_map.get("kernels")
    cycles_field = field_map.get("compute_cycles")
    for row in reader:
        layer = (row.get(field_map["layer"]) or "").strip()
        if not layer:
            continue
        kernels_text = row.get(kernels_field, "") if kernels_field else ""
        cycles_text = row.get(cycles_field, "") if cycles_field else ""
        entry = {
            "layer": layer,
            "type": (row.get(field_map["type"]) or "").strip(),
            "kernels": [
                part.strip() for part in kernels_text.splitlines() if part.strip()
            ],
            "tag": ";".join(
                part.strip() for part in kernels_text.splitlines() if part.strip()
            ),
            "placement": (row.get(field_map["placement"]) or "").strip(),
            "time_us": _parse_timing_us(row.get(field_map["timing"]) or "0 us"),
            "cycles": _parse_int(cycles_text) if cycles_text.strip() else None,
        }
        entry["is_delegate"] = str(entry["placement"]).startswith("NPU")
        entry["is_fallback"] = str(entry["placement"]).startswith("CPU")
        entries.append(entry)

    if not entries:
        return None
    return {
        "job_id": job_id,
        "source": "runtime_analysis",
        "entries": entries,
        "summary": summarize_layer_entries(entries),
    }


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
