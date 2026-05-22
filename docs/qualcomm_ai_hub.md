# Qualcomm AI Hub Profiling

## Why This Exists

The local TensorFlow Lite CPU benchmark in this repository is a reproducibility proxy. It measures inference speed on the developer machine, not on an Android phone and not inside the full app loop.

Qualcomm AI Hub can profile exported TFLite HAR models on cloud-hosted Qualcomm devices. This helps test whether the lightweight WISDM and UCI HAR models fit mobile and edge deployment constraints before full Android app integration.

For tiny HAR models, CPU inference may already be sufficient. NPU profiling is still useful because it can reveal dispatch overhead, unsupported operator fallback, and whether acceleration is actually worthwhile for small motion-signal models.

## Setup

Install the optional Qualcomm dependencies:

```bash
uv pip install qai-hub python-dotenv
```

Or install the project extra:

```bash
uv sync --extra qualcomm
```

Put the API token in a local secret file that is ignored by git:

```text
.secrets/.env
```

Example:

```text
QUALCOMM_AI_HUB_API_KEY=your_token_here
```

Do not commit `.secrets/`, `.env`, profile outputs, compiled assets, model binaries, datasets, or benchmark artifacts.

Check the environment without submitting jobs:

```bash
uv run --extra qualcomm python scripts/qaihub_check.py --list-devices
```

Optional persistent configure:

```bash
uv run --extra qualcomm python scripts/qaihub_configure_from_env.py --persist
```

This writes the token to the user's local Qualcomm AI Hub config, not to the repository.

## Profile UCI HAR TinyTCN

CPU profile:

```bash
uv run --extra qualcomm python scripts/profile_qualcomm_ai_hub.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-unit cpu \
  --wait \
  --artifacts-dir outputs/qualcomm_ai_hub/artifacts
```

NPU profile:

```bash
uv run --extra qualcomm python scripts/profile_qualcomm_ai_hub.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-unit npu \
  --wait \
  --artifacts-dir outputs/qualcomm_ai_hub/artifacts
```

## Repeated Profiles

Run repeated profiles to measure run-to-run variance:

```bash
uv run --extra qualcomm python scripts/run_qaihub_repeated_profiles.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-units cpu npu \
  --runs 5 \
  --wait
```

The runner writes one summary JSON per run plus `manifest.json` and `manifest.csv` under `outputs/qualcomm_ai_hub/repeated/`.

Use dry-run mode to validate paths and arguments without submitting jobs:

```bash
uv run python scripts/run_qaihub_repeated_profiles.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --compute-units cpu npu \
  --runs 1 \
  --dry-run
```

## Structured Metric Export

Export clean JSON/CSV metrics from downloaded or manually saved runtime logs:

```bash
uv run python scripts/export_qaihub_profile_metrics.py \
  --logs j5wm0e06g_runtime.log jp3qljll5_runtime.log \
  --output-json outputs/qualcomm_ai_hub/profile_metrics.json \
  --output-csv outputs/qualcomm_ai_hub/profile_metrics.csv \
  --layer-csv outputs/qualcomm_ai_hub/profile_layers.csv
```

The export converts log memory ranges from kB to MB and keeps the original range semantics:

- `increase_min_mb` / `increase_max_mb`: memory increase range reported by the profiler.
- `peak_delta_min_mb` / `peak_delta_max_mb`: peak range reported by the profiler.
- `cold_load`, `warm_load`, `inference`, and `by_layer` phases are kept separate.

If Qualcomm Workbench shows the Runtime Analysis table, copy it into a text file or pipe it through stdin:

```bash
uv run python scripts/export_qaihub_profile_metrics.py \
  --logs jp3qljll5_runtime.log \
  --runtime-analysis jp3qljll5=runtime_analysis_copy.txt
```

The Runtime Analysis source gives the cleanest op placement breakdown because it includes `NPU (QNN)` versus `CPU (TfLite)` placement per layer.

## Numeric Parity

Compare local TensorFlow Lite output with AI Hub inference output for a deterministic synthetic input:

```bash
uv run --extra qualcomm python scripts/qaihub_numeric_parity.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-unit cpu \
  --wait \
  --output-json outputs/qualcomm_ai_hub/numeric_parity_cpu.json
```

Run the same check for the requested NPU path:

```bash
uv run --extra qualcomm python scripts/qaihub_numeric_parity.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-unit npu \
  --wait \
  --output-json outputs/qualcomm_ai_hub/numeric_parity_npu.json
```

The default tolerance is `atol=1e-4`, `rtol=1e-4`. CPU should normally be near bit-level parity. NPU/QNN can have larger floating-point differences, so report both strict allclose status and task-level behavior such as top-class match.

## V5 CPU/GPU/NPU Complexity Sweep

Train, export, benchmark, and audit the three required UCI HAR models:

```bash
uv run python scripts/train_har.py --config configs/uci_har_tinytcn_v5.yaml
uv run python scripts/train_har.py --config configs/uci_har_tiny_cnn1d.yaml
uv run python scripts/train_har.py --config configs/uci_har_medium_conv1d.yaml

uv run python scripts/export_tflite.py --model-dir outputs/v5/uci_har_tinytcn --output outputs/v5/uci_har_tinytcn/model.tflite
uv run python scripts/export_tflite.py --model-dir outputs/v5/uci_har_tiny_cnn1d --output outputs/v5/uci_har_tiny_cnn1d/model.tflite
uv run python scripts/export_tflite.py --model-dir outputs/v5/uci_har_medium_conv1d --output outputs/v5/uci_har_medium_conv1d/model.tflite

uv run python scripts/benchmark_tflite.py --model outputs/v5/uci_har_tinytcn/model.tflite --input-shape 1,128,6 --warmup 50 --runs 1000 --seed 20260521 --output outputs/v5/uci_har_tinytcn/local_benchmark.json
uv run python scripts/benchmark_tflite.py --model outputs/v5/uci_har_tiny_cnn1d/model.tflite --input-shape 1,128,6 --warmup 50 --runs 1000 --seed 20260521 --output outputs/v5/uci_har_tiny_cnn1d/local_benchmark.json
uv run python scripts/benchmark_tflite.py --model outputs/v5/uci_har_medium_conv1d/model.tflite --input-shape 1,128,6 --warmup 50 --runs 1000 --seed 20260521 --output outputs/v5/uci_har_medium_conv1d/local_benchmark.json

uv run python scripts/op_audit_tflite.py --model outputs/v5/uci_har_tinytcn/model.tflite --output outputs/v5/uci_har_tinytcn/op_audit.json
uv run python scripts/op_audit_tflite.py --model outputs/v5/uci_har_tiny_cnn1d/model.tflite --output outputs/v5/uci_har_tiny_cnn1d/op_audit.json
uv run python scripts/op_audit_tflite.py --model outputs/v5/uci_har_medium_conv1d/model.tflite --output outputs/v5/uci_har_medium_conv1d/op_audit.json
```

Run the full Qualcomm matrix:

```bash
uv run --extra qualcomm python scripts/run_qaihub_repeated_profiles.py \
  --models \
    outputs/v5/uci_har_tinytcn/model.tflite \
    outputs/v5/uci_har_tiny_cnn1d/model.tflite \
    outputs/v5/uci_har_medium_conv1d/model.tflite \
  --model-names tinytcn tiny_cnn1d medium_conv1d \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-units cpu,gpu,npu \
  --input-shape 1,128,6 \
  --runs 5 \
  --wait \
  --artifacts-dir outputs/qualcomm_ai_hub/v5/artifacts_real \
  --out-dir outputs/qualcomm_ai_hub/v5/repeated_real
```

Aggregate repeated profile summaries and generate the v5 report:

```bash
uv run python scripts/aggregate_qaihub_profiles.py \
  --input-dir outputs/qualcomm_ai_hub/v5/repeated_real \
  --output outputs/qualcomm_ai_hub/v5/aggregate_summary.json \
  --markdown outputs/qualcomm_ai_hub/v5/aggregate_summary.md

uv run python scripts/generate_v5_report.py \
  --local-results outputs/v5 \
  --qualcomm-summary outputs/qualcomm_ai_hub/v5/aggregate_summary.json \
  --parity-results outputs/qualcomm_ai_hub/v5/parity_real \
  --output reports/report_v5_cpu_gpu_npu_complexity_sweep.md
```

Profile summaries parse downloaded runtime logs when available and check downloaded profile artifacts for energy/power fields. If the device tooling does not expose numeric energy or power values, the report must mark those fields as missing/not exposed rather than estimating them.

The report must distinguish local CPU TFLite latency, AI Hub hosted-device profile status, Workbench/runtime-log metrics, energy/power availability, and Android app end-to-end latency.

## Profile UCI HAR 1D-CNN Fallback

```bash
uv run --extra qualcomm python scripts/profile_qualcomm_ai_hub.py \
  --model outputs/uci_har_cnn1d/model.tflite \
  --input-shape 1,128,6 \
  --device "Samsung Galaxy S24 (Family)" \
  --compute-unit cpu \
  --wait
```

## Manual Commands

Print CLI commands without reading or printing the real API token:

```bash
uv run python scripts/print_qaihub_commands.py \
  --model outputs/uci_har_tinytcn/model.tflite \
  --device "Samsung Galaxy S24 (Family)"
```

## Interpreting Results

Qualcomm AI Hub profile latency is not Android app end-to-end latency. App latency also includes sensor collection, window buffering, preprocessing, UI work, and OS scheduling.

NPU profiling may not be faster for tiny models because dispatch overhead can dominate. NPU jobs can also fail or fall back if a runtime or operator is unsupported. If that happens, document the result directly instead of treating it as a model failure.

Profile summaries are written under:

```text
outputs/qualcomm_ai_hub/
```

Those files are local artifacts and must remain untracked.

## Result Table Template

| Dataset | Model | Device | Runtime | Compute unit | Latency ms | Memory MB | Energy mJ | Power mW | Status | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| UCI HAR | TinyTCN | pending | TFLite | CPU | pending | pending | pending | pending | pending | Qualcomm AI Hub |
| UCI HAR | TinyTCN | pending | TFLite | NPU | pending | pending | pending | pending | pending | Qualcomm AI Hub |
