#!/usr/bin/env python3
"""Run repeated Qualcomm AI Hub profile jobs and collect summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_qualcomm_ai_hub.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/uci_har_tinytcn/model.tflite")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of model paths for matrix profiling.",
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="Names matching --models. Defaults to model parent directory names.",
    )
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["cpu", "npu"],
        help="Compute units as space-separated values or comma-separated groups.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--out-dir", default="outputs/qualcomm_ai_hub/repeated")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "Optional base directory for downloaded AI Hub profile artifacts. "
            "Each run is stored under model/unit/run/job_id."
        ),
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs <= 0:
        print("--runs must be positive")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_specs = _model_specs(args)
    compute_units = _parse_compute_units(args.compute_units)
    rows: list[dict[str, Any]] = []
    return_code = 0

    for model_name, model_path in model_specs:
        for compute_unit in compute_units:
            for run_index in range(1, args.runs + 1):
                output_json = (
                    out_dir / model_name / compute_unit / f"run_{run_index:02d}.json"
                )
                command = [
                    sys.executable,
                    str(PROFILE_SCRIPT),
                    "--model",
                    str(model_path),
                    "--input-shape",
                    args.input_shape,
                    "--device",
                    args.device,
                    "--compute-unit",
                    compute_unit,
                    "--env-file",
                    args.env_file,
                    "--output-json",
                    str(output_json),
                ]
                if args.wait:
                    command.append("--wait")
                if args.dry_run:
                    command.append("--dry-run")
                if args.artifacts_dir:
                    artifacts_dir = (
                        Path(args.artifacts_dir)
                        / model_name
                        / compute_unit
                        / f"run_{run_index:02d}"
                    )
                    command.extend(["--artifacts-dir", str(artifacts_dir)])

                print(
                    f"[{model_name} {compute_unit} run {run_index}/{args.runs}] "
                    "submitting profile"
                )
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    print(result.stdout.strip())
                if result.stderr:
                    print(result.stderr.strip(), file=sys.stderr)

                row = _load_summary_row(output_json)
                row.update(
                    {
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "run_index": run_index,
                        "compute_unit": compute_unit,
                        "return_code": result.returncode,
                        "summary_json": str(output_json),
                    }
                )
                rows.append(row)

                if result.returncode != 0:
                    return_code = result.returncode
                    if args.stop_on_failure:
                        _write_manifest(out_dir, args, model_specs, compute_units, rows)
                        return return_code

    _write_manifest(out_dir, args, model_specs, compute_units, rows)
    return return_code


def _load_summary_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing_summary", "job_id": None, "job_url": None}
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": summary.get("status"),
        "job_id": summary.get("job_id"),
        "job_url": summary.get("job_url"),
        "latency_ms": summary.get("latency_ms"),
        "memory_mb": summary.get("memory_mb"),
        "energy_mj": summary.get("energy_mj"),
        "power_mw": summary.get("power_mw"),
        "energy_power_available": summary.get("energy_power_available"),
        "runtime_path": summary.get("runtime_path"),
        "delegate": summary.get("delegate"),
        "delegated_nodes": summary.get("delegated_nodes"),
        "total_nodes": summary.get("total_nodes"),
        "fully_delegated": summary.get("fully_delegated"),
        "fallback_ops": summary.get("fallback_ops"),
        "notes": summary.get("notes"),
    }


def _write_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    model_specs: list[tuple[str, Path]],
    compute_units: list[str],
    rows: list[dict[str, Any]],
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": [
            {"model_name": model_name, "model_path": str(model_path)}
            for model_name, model_path in model_specs
        ],
        "input_shape": args.input_shape,
        "device": args.device,
        "compute_units": compute_units,
        "runs": args.runs,
        "wait": args.wait,
        "dry_run": args.dry_run,
        "artifacts_dir": args.artifacts_dir,
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if rows:
        fieldnames = list(rows[0].keys())
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote repeated profile manifest to {out_dir / 'manifest.json'}")


def _model_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    model_paths = [Path(path) for path in (args.models or [args.model])]
    if args.model_names is not None and len(args.model_names) != len(model_paths):
        raise SystemExit("--model-names must have the same length as --models")

    names = args.model_names or [path.parent.name or path.stem for path in model_paths]
    return [(name, path) for name, path in zip(names, model_paths, strict=True)]


def _parse_compute_units(values: list[str]) -> list[str]:
    allowed = {"cpu", "gpu", "npu", "all", "auto"}
    units: list[str] = []
    for value in values:
        for unit in value.split(","):
            unit = unit.strip().lower()
            if not unit:
                continue
            if unit not in allowed:
                raise SystemExit(
                    f"Unsupported compute unit {unit!r}; expected one of {sorted(allowed)}"
                )
            units.append(unit)
    return units


if __name__ == "__main__":
    raise SystemExit(main())
