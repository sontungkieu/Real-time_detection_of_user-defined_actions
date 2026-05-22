#!/usr/bin/env python3
"""Profile an exported TFLite HAR model with optional Qualcomm AI Hub support."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.utils.secrets import get_qualcomm_api_key

try:
    from activity_recognition.utils.qaihub_metrics import parse_runtime_log
except Exception:  # pragma: no cover - profiling should still submit without parser.
    parse_runtime_log = None

INSTALL_MESSAGE = "Install optional dependency: uv pip install qai-hub python-dotenv"
CONFIGURE_MESSAGE = "Run: qai-hub configure --api_token <token>"
STATUS_VALUES = {"success", "failed", "pending"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/uci_har_tinytcn/model.tflite")
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    parser.add_argument(
        "--compute-unit",
        choices=("cpu", "gpu", "npu", "all", "auto"),
        default="cpu",
    )
    parser.add_argument("--target-runtime", default="tflite")
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/qualcomm_ai_hub/profile_summary.json",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory for AI Hub profile JSON and job logs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    output_json = Path(args.output_json)

    try:
        input_shape = _parse_input_shape(args.input_shape)
        _validate_model_path(model_path)
    except ValueError as exc:
        print(str(exc))
        summary = _base_summary(args, model_path, output_json, None, "failed", str(exc))
        _save_summary(summary, output_json)
        return 1

    if args.dry_run:
        summary = _base_summary(
            args,
            model_path,
            output_json,
            input_shape,
            "pending",
            "dry-run only; no Qualcomm AI Hub job was submitted",
        )
        summary["dry_run"] = True
        _save_summary(summary, output_json)
        print(f"Dry run passed. Summary written to {output_json}")
        return 0

    api_key = get_qualcomm_api_key(args.env_file)
    if not api_key:
        message = f"Missing QUALCOMM_AI_HUB_API_KEY in {args.env_file}"
        print(message)
        _save_summary(
            _base_summary(
                args, model_path, output_json, input_shape, "failed", message
            ),
            output_json,
        )
        return 1

    try:
        import qai_hub as hub
    except ImportError:
        print(INSTALL_MESSAGE)
        _save_summary(
            _base_summary(
                args, model_path, output_json, input_shape, "failed", INSTALL_MESSAGE
            ),
            output_json,
        )
        return 1

    summary, return_code = _submit_with_library(
        hub, args, model_path, input_shape, output_json, api_key
    )
    if summary is None:
        summary, return_code = _submit_with_cli(
            args, model_path, input_shape, output_json
        )

    _save_summary(summary, output_json)
    print(f"Qualcomm AI Hub profile summary written to {output_json}")
    return return_code


def _submit_with_library(
    hub,
    args: argparse.Namespace,
    model_path: Path,
    input_shape: list[int],
    output_json: Path,
    api_key: str,
) -> tuple[dict[str, Any] | None, int]:
    try:
        client_config = hub.ClientConfig(api_token=api_key)
        client = hub.Client(client_config)
        device = hub.Device(args.device)
    except (AttributeError, TypeError):
        return None, 1
    except Exception as exc:
        message = (
            "Unable to initialize Qualcomm AI Hub session client: "
            f"{type(exc).__name__}: {_redact_secret(str(exc), api_key)}. "
            f"{CONFIGURE_MESSAGE}"
        )
        return (
            _base_summary(
                args, model_path, output_json, input_shape, "failed", message
            ),
            1,
        )

    try:
        job = client.submit_profile_job(
            model=model_path,
            device=device,
            options=_profile_options(args.compute_unit),
            name=f"HAR {model_path.name} {args.compute_unit}",
        )
    except (AttributeError, TypeError):
        return None, 1
    except Exception as exc:
        message = (
            "Qualcomm AI Hub library profile submission failed: "
            f"{type(exc).__name__}: {_redact_secret(str(exc), api_key)}. "
            "Trying CLI fallback if available."
        )
        print(message)
        return None, 1

    status = "pending"
    if args.wait:
        try:
            status = _normalize_status(job.wait())
        except Exception as exc:
            message = f"Profile job wait failed: {type(exc).__name__}: {exc}"
            summary = _base_summary(
                args, model_path, output_json, input_shape, "failed", message
            )
            _add_job_identifiers(summary, job)
            return summary, 1
    else:
        try:
            status = _normalize_status(job.get_status())
        except Exception:
            status = "pending"

    summary = _base_summary(
        args,
        model_path,
        output_json,
        input_shape,
        status,
        "submitted with Qualcomm AI Hub Python client",
    )
    _add_job_identifiers(summary, job)

    if args.wait and status == "success":
        _download_job_artifacts(summary, job, args.artifacts_dir)
        summary["notes"] += (
            "; export structured metrics from downloaded/runtime logs with "
            "scripts/export_qaihub_profile_metrics.py"
        )

    return summary, 0 if status in {"success", "pending"} else 1


def _submit_with_cli(
    args: argparse.Namespace,
    model_path: Path,
    input_shape: list[int],
    output_json: Path,
) -> tuple[dict[str, Any], int]:
    executable = shutil.which("qai-hub")
    if executable is None:
        message = f"{INSTALL_MESSAGE}. CLI fallback unavailable."
        return (
            _base_summary(
                args, model_path, output_json, input_shape, "failed", message
            ),
            1,
        )

    command = [
        executable,
        "submit-profile-job",
        "--model",
        str(model_path),
        "--device",
        args.device,
    ]
    profile_options = _profile_options(args.compute_unit)
    if profile_options:
        command.extend(["--profile_options", profile_options])
    if args.wait:
        command.append("--wait")

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    status = "success" if result.returncode == 0 else "failed"
    summary = _base_summary(
        args,
        model_path,
        output_json,
        input_shape,
        status,
        "submitted with qai-hub CLI fallback; persistent local configure may be required",
    )
    summary["job_id"] = _extract_job_id(combined_output)
    summary["job_url"] = _extract_url(combined_output)
    if result.returncode != 0:
        summary["error"] = _safe_cli_error(combined_output)
        summary["suggested_next_action"] = CONFIGURE_MESSAGE
    return summary, result.returncode


def _base_summary(
    args: argparse.Namespace,
    model_path: Path,
    output_json: Path,
    input_shape: list[int] | None,
    status: str,
    notes: str,
) -> dict[str, Any]:
    normalized_status = status if status in STATUS_VALUES else "pending"
    return {
        "model": str(model_path),
        "input_shape": input_shape,
        "device": args.device,
        "target_runtime": args.target_runtime,
        "compute_unit_requested": args.compute_unit,
        "job_id": None,
        "job_url": None,
        "status": normalized_status,
        "latency_ms": None,
        "memory_mb": None,
        "energy_mj": None,
        "power_mw": None,
        "energy_power_available": None,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_json": str(output_json),
    }


def _parse_input_shape(value: str) -> list[int]:
    try:
        shape = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"Invalid --input-shape {value!r}; expected comma-separated integers."
        ) from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(
            f"Invalid --input-shape {value!r}; all dimensions must be positive."
        )
    return shape


def _validate_model_path(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Model file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Model path is not a file: {path}")


def _profile_options(compute_unit: str) -> str:
    if compute_unit == "auto":
        return ""
    return f" --compute_unit {compute_unit}"


def _normalize_status(status: object) -> str:
    text = str(getattr(status, "name", status)).lower()
    if "success" in text or "completed" in text:
        return "success"
    if "fail" in text or "error" in text or "cancel" in text:
        return "failed"
    return "pending"


def _add_job_identifiers(summary: dict[str, Any], job: object) -> None:
    summary["job_id"] = _first_attr(job, "job_id", "id", "_job_id")
    summary["job_url"] = _first_attr(job, "url", "web_url", "_url")


def _first_attr(obj: object, *names: str) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return str(value)
    return None


def _download_job_artifacts(
    summary: dict[str, Any], job: object, artifacts_dir: str | None
) -> None:
    if not artifacts_dir:
        return

    job_id = summary.get("job_id") or "profile_job"
    output_dir = Path(artifacts_dir) / str(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Any] = {}

    try:
        profile_path = output_dir / "profile.json"
        result = job.download_profile(filename=str(profile_path))
        downloaded["profile"] = str(result if isinstance(result, str) else profile_path)
    except Exception as exc:
        downloaded["profile_error"] = f"{type(exc).__name__}: {exc}"

    try:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        downloaded["logs"] = job.download_job_logs(str(logs_dir))
    except Exception as exc:
        downloaded["logs_error"] = f"{type(exc).__name__}: {exc}"

    summary["downloaded_artifacts"] = downloaded
    summary["artifacts_dir"] = str(output_dir)
    _populate_summary_from_artifacts(summary, output_dir)


def _populate_summary_from_artifacts(summary: dict[str, Any], output_dir: Path) -> None:
    runtime_records = []
    if parse_runtime_log is not None:
        for path in _iter_artifact_files(output_dir):
            if path.suffix.lower() not in {".log", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "Tungsten" not in text and "Performed Inference" not in text:
                continue
            try:
                record = parse_runtime_log(path)
            except Exception as exc:
                summary.setdefault("artifact_parse_errors", []).append(
                    f"{path}: {type(exc).__name__}: {exc}"
                )
                continue
            runtime_records.append(record)
            _apply_runtime_metrics(summary, record)

    energy_power_metrics: list[dict[str, Any]] = []
    for path in _iter_artifact_files(output_dir):
        if path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        energy_power_metrics.extend(_extract_energy_power_metrics(data))

    summary["runtime_log_records"] = len(runtime_records)
    summary["energy_power_available"] = bool(energy_power_metrics)
    if energy_power_metrics:
        summary["energy_power_metrics"] = energy_power_metrics[:50]
        _apply_energy_power_candidates(summary, energy_power_metrics)
    else:
        summary["energy_power_notes"] = (
            "No energy/power fields were exposed in downloaded AI Hub profile "
            "artifacts for this job."
        )


def _apply_runtime_metrics(summary: dict[str, Any], record: dict[str, Any]) -> None:
    timings = record.get("timings", {})
    memory = record.get("memory", {})
    delegate = record.get("delegate", {})
    by_layer_summary = record.get("by_layer", {}).get("summary", {})

    inference_us = timings.get("inference_us")
    if isinstance(inference_us, int | float):
        summary["latency_ms"] = round(float(inference_us) / 1000.0, 6)
    cold_load_us = timings.get("cold_load_us")
    if isinstance(cold_load_us, int | float):
        summary["cold_load_ms"] = round(float(cold_load_us) / 1000.0, 6)
    warm_mean_us = timings.get("warm_load_us_mean")
    if isinstance(warm_mean_us, int | float):
        summary["warm_load_mean_ms"] = round(float(warm_mean_us) / 1000.0, 6)
    by_layer_us = timings.get("by_layer_us")
    if isinstance(by_layer_us, int | float):
        summary["by_layer_ms"] = round(float(by_layer_us) / 1000.0, 6)

    inference_memory = memory.get("inference") or {}
    if isinstance(inference_memory, dict):
        summary["memory_mb"] = inference_memory.get("increase_max_mb")
        summary["memory_increase_min_mb"] = inference_memory.get("increase_min_mb")
        summary["memory_increase_max_mb"] = inference_memory.get("increase_max_mb")

    summary["runtime_path"] = record.get("compute_unit_observed")
    summary["delegate"] = delegate.get("name")
    summary["delegate_type"] = delegate.get("delegate_type")
    summary["delegated_nodes"] = delegate.get("delegated_nodes")
    summary["total_nodes"] = delegate.get("total_nodes")
    summary["delegate_partitions"] = delegate.get("partitions")
    summary["fully_delegated"] = delegate.get("fully_delegated")
    summary["fallback_entries"] = by_layer_summary.get("fallback_entries")
    summary["fallback_ops"] = by_layer_summary.get("fallback_ops") or {}


def _extract_energy_power_metrics(data: Any) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if len(metrics) >= 200:
            return
        if isinstance(value, dict):
            unit = _first_text_value(value, "unit", "units")
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if _is_number(item) and _looks_like_energy_power(child_path):
                    metrics.append(
                        _energy_power_entry(child_path, float(item), unit=unit)
                    )
                walk(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(data, "")
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for metric in metrics:
        key = (metric["path"], metric["value"])
        if key not in seen:
            seen.add(key)
            deduped.append(metric)
    return deduped


def _apply_energy_power_candidates(
    summary: dict[str, Any], metrics: list[dict[str, Any]]
) -> None:
    for metric in metrics:
        if summary.get("energy_mj") is None and metric.get("kind") == "energy":
            converted = metric.get("value_mj")
            if converted is not None:
                summary["energy_mj"] = converted
        if summary.get("power_mw") is None and metric.get("kind") == "power":
            converted = metric.get("value_mw")
            if converted is not None:
                summary["power_mw"] = converted


def _energy_power_entry(path: str, value: float, unit: str | None) -> dict[str, Any]:
    lower_path = path.lower()
    lower_unit = (unit or "").lower()
    kind = "energy" if "energy" in lower_path else "power"
    entry: dict[str, Any] = {
        "path": path,
        "kind": kind,
        "value": value,
        "unit": unit,
    }
    if kind == "energy":
        entry["value_mj"] = _convert_energy_to_mj(value, lower_path, lower_unit)
    else:
        entry["value_mw"] = _convert_power_to_mw(value, lower_path, lower_unit)
    return entry


def _convert_energy_to_mj(
    value: float, lower_path: str, lower_unit: str
) -> float | None:
    token = f"{lower_path} {lower_unit}"
    if "uj" in token or "microjoule" in token:
        return value / 1000.0
    if "mj" in token or "millijoule" in token:
        return value
    if re.search(r"(^|[^a-z])j(oule)?s?([^a-z]|$)", token):
        return value * 1000.0
    return None


def _convert_power_to_mw(
    value: float, lower_path: str, lower_unit: str
) -> float | None:
    token = f"{lower_path} {lower_unit}"
    if "uw" in token or "microwatt" in token:
        return value / 1000.0
    if "mw" in token or "milliwatt" in token:
        return value
    if re.search(r"(^|[^a-z])w(att)?s?([^a-z]|$)", token):
        return value * 1000.0
    return None


def _looks_like_energy_power(path: str) -> bool:
    lower = path.lower()
    return "energy" in lower or "power" in lower


def _first_text_value(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _iter_artifact_files(output_dir: Path) -> list[Path]:
    return [path for path in sorted(output_dir.rglob("*")) if path.is_file()]


def _extract_job_id(output: str) -> str | None:
    match = re.search(r"\bj[a-z0-9]{6,}\b", output, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _extract_url(output: str) -> str | None:
    match = re.search(r"https?://\S+", output)
    return match.group(0).rstrip(".,)") if match else None


def _safe_cli_error(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        return "qai-hub CLI profile submission failed without output."
    return stripped[-1000:]


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "<token>") if secret else text


def _save_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
