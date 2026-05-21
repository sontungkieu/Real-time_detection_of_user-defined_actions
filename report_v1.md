# Report v1: WISDM Benchmark Pipeline Results

Date: 2026-05-21

## Summary

This report documents the first real WISDM run for the project and records the implementation decisions that went beyond the original `plan.md` scope.

The repository now has a reproducible Python pipeline for:

- loading WISDM v1.1 raw accelerometer data,
- creating sliding windows,
- splitting by subject to reduce leakage,
- training a lightweight 1D-CNN,
- evaluating classification metrics,
- exporting TensorFlow Lite,
- benchmarking local CPU TFLite latency.

The original prototype scripts remain available at the repository root.

## Dataset

The official Fordham WISDM endpoint timed out from this environment. To complete the run, the dataset was downloaded from the Google Drive mirror referenced by the Curiousily WISDM tutorial:

```bash
uvx gdown 152sWECukjvLerrVG2NUO8gtMFg83RKCF -O data/raw/wisdm/WISDM_ar_latest.tar.gz
tar -xzf data/raw/wisdm/WISDM_ar_latest.tar.gz -C data/raw/wisdm
```

Loaded raw file:

```text
data/raw/wisdm/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt
```

Parsed dataset stats:

| Item | Value |
| --- | ---: |
| Raw rows parsed | 1,098,199 |
| Subjects | 36 |
| Activities | 6 |
| Sliding windows | 16,890 |
| Window shape | 128 x 4 |

The four channels are `x`, `y`, `z`, and magnitude.

## Split

The run used a subject-wise split from `configs/wisdm_cnn1d.yaml`.

| Split | Subjects | Windows |
| --- | ---: | ---: |
| Train | 25 | 11,528 |
| Validation | 5 | 2,476 |
| Test | 6 | 2,886 |

Test subjects:

```text
22, 8, 11, 10, 21, 17
```

## Model

Model: lightweight 1D-CNN

Architecture summary:

- Conv1D 32 filters
- Conv1D 64 filters
- GlobalAveragePooling1D
- Dense 64
- Dropout
- Softmax output

| Item | Value |
| --- | ---: |
| Parameters | 11,430 |
| Epochs | 30 |
| Final train accuracy | 0.986 |
| Final validation accuracy | 0.686 |
| Best validation accuracy | 0.791 at epoch 1 |

The validation curve suggests overfitting after early epochs. Future work should add early stopping or checkpoint best validation weights.

## Classification Results

Evaluation was run on held-out subjects only.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.787 |
| Macro-F1 | 0.792 |
| Weighted-F1 | 0.801 |
| Test windows | 2,886 |

Per-class results:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Downstairs | 0.380 | 0.505 | 0.434 | 323 |
| Jogging | 0.996 | 0.941 | 0.968 | 828 |
| Sitting | 0.985 | 1.000 | 0.992 | 65 |
| Standing | 0.867 | 1.000 | 0.929 | 117 |
| Upstairs | 0.520 | 0.656 | 0.580 | 445 |
| Walking | 0.939 | 0.773 | 0.848 | 1,108 |

Main observation:

- Jogging, Sitting, and Standing perform strongly.
- Upstairs and Downstairs are much harder and likely confused with each other or with Walking.
- Walking has high precision but lower recall, meaning the model misses a meaningful number of true Walking windows.

## TensorFlow Lite Results

Exported model:

```text
outputs/wisdm_cnn1d/model.tflite
```

| Metric | Value |
| --- | ---: |
| Keras model size | 0.164 MB |
| TFLite model size | 0.048 MB / 49.28 KB |
| Benchmark input shape | 1 x 128 x 4 |
| Warmup runs | 50 |
| Measured runs | 500 |
| Mean latency | 0.0231 ms |
| Median latency | 0.0220 ms |
| P95 latency | 0.0242 ms |

This benchmark was measured on local CPU with TensorFlow Lite XNNPACK. It is not Android device latency.

Local machine:

```text
Intel Core i5-9300H CPU @ 2.40GHz
Python 3.11.6
TensorFlow 2.21.0
```

## Outputs Generated

Generated files are intentionally ignored by git.

```text
outputs/wisdm_cnn1d/model.keras
outputs/wisdm_cnn1d/model.tflite
outputs/wisdm_cnn1d/metrics.json
outputs/wisdm_cnn1d/classification_report.txt
outputs/wisdm_cnn1d/confusion_matrix.png
outputs/wisdm_cnn1d/benchmark.json
outputs/wisdm_cnn1d/history.csv
```

## Decisions Outside the Original Plan

### 1. Added `pyproject.toml` and `uv.lock`

The original plan only asked to create or update `requirements.txt`. The user later requested `uv add` and `pyproject.toml`, so dependency management was upgraded to a uv-managed project.

Reason:

- makes environment setup reproducible,
- records exact dependency resolution in `uv.lock`,
- supports `uv sync` and `uv run` workflows.

Impact:

- `requirements.txt` remains as a fallback,
- `pyproject.toml` is now the primary dependency source.

### 2. Used a Google Drive WISDM mirror

The original plan preferred direct WISDM download if stable, otherwise instructions. The official Fordham URL repeatedly timed out from this environment, so a Google Drive mirror referenced by an external WISDM tutorial was used.

Reason:

- needed to complete a real WISDM run,
- mirror provided the expected `WISDM_ar_v1.1_raw.txt` archive.

Impact:

- README now documents the fallback `uvx gdown` command,
- dataset remains ignored and is not committed.

### 3. Ran a synthetic sanity run before real WISDM

Before the real WISDM archive was available, a small WISDM-like synthetic run was created to verify training, evaluation, export, and benchmark paths.

Reason:

- confirmed the pipeline was functional while the official endpoint was unavailable.

Impact:

- synthetic data and outputs are ignored,
- synthetic metrics were replaced in README by real WISDM results once the dataset was obtained.

### 4. Expanded `.gitignore` beyond the minimal plan entries

The plan listed dataset/model/output ignores. The final `.gitignore` also includes broader Python project patterns.

Reason:

- avoid committing bytecode, test caches, build artifacts, notebook caches, virtualenvs, local secrets, IDE files, archives, and generated ML artifacts.

Impact:

- safer Python repository hygiene,
- `plan.md` remains local-only as requested.

### 5. Removed `docs/personalization_plan.md` from the recent commits

The original plan asked for `docs/personalization_plan.md`, but the user later requested that it be removed from the last two commits along with `plan.md`.

Reason:

- user explicitly requested exclusion.

Impact:

- personalization is still mentioned in README,
- no separate `docs/personalization_plan.md` is tracked.

### 6. Split the previous work into two commits

The original request asked for a detailed commit. After correction, the last two commits became:

```text
2427bc0 feat(pipeline): add WISDM benchmark workflow
cbc2402 docs(readme): document benchmark workflow
```

Reason:

- separate implementation from documentation,
- remove unwanted files from commit history.

Impact:

- clearer review history,
- `plan.md` and `docs/personalization_plan.md` are excluded.

### 7. Used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for tests

The environment auto-loaded unrelated ROS pytest plugins, causing a missing `lark` dependency error.

Reason:

- isolate this repo's tests from globally installed pytest plugins.

Impact:

- README documents the safer test command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

### 8. Did not update PDF or milestones

The repo does not currently contain `pdf/`, `milestones.md`, `VERSION`, or `versioning.py`.

Reason:

- required files/mechanisms do not exist in this repository.

Impact:

- PDF rebuild, version bump, release notes, and milestones are N/A for this task.

## Risks and Follow-up Work

Recommended next steps:

- add early stopping and save best validation checkpoint,
- run MLP baseline for comparison,
- optionally add a balanced or class-weighted training option,
- inspect confusion matrix and report activity-pair confusions explicitly,
- run repeated seeds to reduce single-split variance,
- measure TFLite latency on an actual Android device,
- add an official documented dataset checksum after download.

## Current Git/Artifact Policy

Committed source/docs should include code, configs, README, docs, tests, and project metadata.

Ignored local artifacts include:

- `data/raw/`,
- `data/processed/`,
- `outputs/`,
- `/models/`,
- `checkpoints/`,
- `.venv/`,
- `plan.md`,
- model binaries such as `.keras`, `.h5`, `.tflite`.
