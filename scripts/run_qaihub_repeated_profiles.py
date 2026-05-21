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
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        choices=("cpu", "gpu", "npu", "auto"),
        default=["cpu", "npu"],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--out-dir", default="outputs/qualcomm_ai_hub/repeated")
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
    rows: list[dict[str, Any]] = []
    return_code = 0

    for compute_unit in args.compute_units:
        for run_index in range(1, args.runs + 1):
            output_json = out_dir / f"{compute_unit}_run_{run_index:02d}.json"
            command = [
                sys.executable,
                str(PROFILE_SCRIPT),
                "--model",
                args.model,
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

            print(f"[{compute_unit} run {run_index}/{args.runs}] submitting profile")
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
                    _write_manifest(out_dir, args, rows)
                    return return_code

    _write_manifest(out_dir, args, rows)
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
        "notes": summary.get("notes"),
    }


def _write_manifest(
    out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "input_shape": args.input_shape,
        "device": args.device,
        "compute_units": args.compute_units,
        "runs": args.runs,
        "wait": args.wait,
        "dry_run": args.dry_run,
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


if __name__ == "__main__":
    raise SystemExit(main())
