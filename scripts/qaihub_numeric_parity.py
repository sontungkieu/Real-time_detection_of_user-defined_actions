#!/usr/bin/env python3
"""Compare local TFLite output with Qualcomm AI Hub inference output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.utils.secrets import get_qualcomm_api_key  # noqa: E402


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
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument(
        "--output-json",
        default="outputs/qualcomm_ai_hub/numeric_parity.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    output_json = Path(args.output_json)

    if not model_path.exists():
        _save_json(
            _base_payload(
                args, model_path, "failed", f"Model file not found: {model_path}"
            ),
            output_json,
        )
        print(f"Model file not found: {model_path}")
        return 1

    try:
        input_shape = _parse_shape(args.input_shape)
        local = _run_local_tflite(model_path, input_shape, args.seed)
    except Exception as exc:
        payload = _base_payload(
            args,
            model_path,
            "failed",
            f"Local TFLite inference failed: {type(exc).__name__}: {exc}",
        )
        _save_json(payload, output_json)
        print(payload["notes"])
        return 1

    payload = _base_payload(args, model_path, "pending", "local TFLite output computed")
    payload.update(
        {
            "input_shape": input_shape,
            "input_name": local["input_name"],
            "output_name_local": local["output_name"],
            "input_sha256": _array_sha256(local["input"]),
            "local_output_sha256": _array_sha256(local["output"]),
            "local_output_shape": list(local["output"].shape),
            "local_top_class": int(np.argmax(local["output"][0])),
            "local_top_score": float(np.max(local["output"][0])),
        }
    )
    if args.include_values:
        payload["local_output"] = local["output"].tolist()

    if args.dry_run:
        payload["notes"] = (
            "dry-run only; local TFLite output computed without AI Hub job"
        )
        _save_json(payload, output_json)
        print(f"Dry run passed. Numeric parity summary written to {output_json}")
        return 0

    api_key = get_qualcomm_api_key(args.env_file)
    if not api_key:
        payload["status"] = "failed"
        payload["notes"] = f"Missing QUALCOMM_AI_HUB_API_KEY in {args.env_file}"
        _save_json(payload, output_json)
        print(payload["notes"])
        return 1

    try:
        import qai_hub as hub
    except ImportError:
        payload["status"] = "failed"
        payload["notes"] = (
            "Install optional dependency: uv pip install qai-hub python-dotenv"
        )
        _save_json(payload, output_json)
        print(payload["notes"])
        return 1

    try:
        ai_hub = _run_ai_hub_inference(
            hub, args, model_path, local["input"], local["input_name"], api_key
        )
    except Exception as exc:
        payload["status"] = "failed"
        payload["notes"] = (
            f"AI Hub inference failed: {type(exc).__name__}: {_redact(str(exc), api_key)}"
        )
        _save_json(payload, output_json)
        print(payload["notes"])
        return 1

    payload.update(ai_hub["metadata"])
    if ai_hub["output"] is None:
        payload["status"] = "pending" if not args.wait else "failed"
        payload["notes"] = ai_hub["metadata"].get(
            "notes", "AI Hub output was not available"
        )
        _save_json(payload, output_json)
        print(f"Numeric parity summary written to {output_json}")
        return 0 if payload["status"] == "pending" else 1

    comparison = _compare_outputs(
        local["output"], ai_hub["output"], args.atol, args.rtol
    )
    payload.update(comparison)
    payload["status"] = "success" if comparison["allclose"] else "failed"
    payload["ai_hub_output_sha256"] = _array_sha256(ai_hub["output"])
    payload["ai_hub_output_shape"] = list(ai_hub["output"].shape)
    payload["ai_hub_top_class"] = int(np.argmax(ai_hub["output"][0]))
    payload["ai_hub_top_score"] = float(np.max(ai_hub["output"][0]))
    payload["top_class_match"] = (
        payload["local_top_class"] == payload["ai_hub_top_class"]
    )
    payload["notes"] = (
        "numeric parity compared local TFLite output against AI Hub output"
    )
    if args.include_values:
        payload["ai_hub_output"] = ai_hub["output"].tolist()

    _save_json(payload, output_json)
    print(f"Numeric parity summary written to {output_json}")
    return 0 if payload["status"] == "success" else 1


def _run_local_tflite(
    model_path: Path, input_shape: list[int], seed: int
) -> dict[str, Any]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    if list(input_details["shape"]) != input_shape:
        interpreter.resize_tensor_input(input_details["index"], input_shape)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    sample = _make_sample(input_shape, input_details["dtype"], seed)
    interpreter.set_tensor(input_details["index"], sample)
    interpreter.invoke()
    output = np.asarray(interpreter.get_tensor(output_details["index"]))
    return {
        "input_name": str(input_details["name"]),
        "output_name": str(output_details["name"]),
        "input": sample,
        "output": output,
    }


def _run_ai_hub_inference(
    hub: Any,
    args: argparse.Namespace,
    model_path: Path,
    sample: np.ndarray,
    input_name: str,
    api_key: str,
) -> dict[str, Any]:
    client = hub.Client(hub.ClientConfig(api_token=api_key))
    job = client.submit_inference_job(
        model=model_path,
        device=hub.Device(args.device),
        inputs={input_name: [sample]},
        options=_runtime_options(args.compute_unit),
        name=f"HAR parity {model_path.name} {args.compute_unit}",
    )

    metadata = {
        "job_id": _first_attr(job, "job_id", "id", "_job_id"),
        "job_url": _first_attr(job, "url", "web_url", "_url"),
        "compute_unit_requested": args.compute_unit,
    }
    if args.wait:
        status = _normalize_status(job.wait())
        metadata["ai_hub_status"] = status
        if status != "success":
            metadata["notes"] = f"AI Hub inference job finished with status={status}"
            return {"metadata": metadata, "output": None}
    else:
        metadata["ai_hub_status"] = _normalize_status(job.get_status())
        metadata["notes"] = (
            "AI Hub inference job submitted; rerun with --wait to download output"
        )
        return {"metadata": metadata, "output": None}

    outputs = job.download_output_data()
    output_name, output = _extract_first_output(outputs)
    metadata["output_name_ai_hub"] = output_name
    return {"metadata": metadata, "output": output}


def _extract_first_output(outputs: Any) -> tuple[str | None, np.ndarray | None]:
    if outputs is None:
        return None, None
    if isinstance(outputs, Mapping):
        for name, values in outputs.items():
            if values:
                return str(name), np.asarray(values[0])
    return None, None


def _compare_outputs(
    local_output: np.ndarray, ai_hub_output: np.ndarray, atol: float, rtol: float
) -> dict[str, Any]:
    diff = np.abs(local_output - ai_hub_output)
    return {
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(
            np.allclose(local_output, ai_hub_output, atol=atol, rtol=rtol)
        ),
        "allclose_at_1e_4": bool(
            np.allclose(local_output, ai_hub_output, atol=1e-4, rtol=1e-4)
        ),
        "allclose_at_1e_3": bool(
            np.allclose(local_output, ai_hub_output, atol=1e-3, rtol=1e-3)
        ),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "median_abs_diff": float(np.median(diff)),
        "p95_abs_diff": float(np.percentile(diff, 95)),
    }


def _make_sample(input_shape: list[int], dtype: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.normal(loc=0.0, scale=1.0, size=input_shape).astype(dtype)
    return rng.integers(low=0, high=255, size=input_shape, dtype=dtype)


def _base_payload(
    args: argparse.Namespace, model_path: Path, status: str, notes: str
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "device": args.device,
        "compute_unit_requested": args.compute_unit,
        "seed": args.seed,
        "status": status,
        "notes": notes,
    }


def _runtime_options(compute_unit: str) -> str:
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


def _first_attr(obj: object, *names: str) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return str(value)
    return None


def _parse_shape(value: str) -> list[int]:
    shape = [int(part.strip()) for part in value.split(",")]
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("--input-shape must contain positive integers")
    return shape


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<token>") if secret else text


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
