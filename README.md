# Personalized On-device Activity Recognition from Wearable-like Motion Signals

This project studies lightweight activity recognition from mobile/wearable-like motion signals. It combines public human activity recognition benchmark evaluation with a personalized user-defined action recognition prototype based on self-collected accelerometer data.

The repository started as an early Python prototype for collecting mobile accelerometer streams, training a small Keras model, and running real-time prediction from a socket connection. The current direction is to keep that prototype accessible while adding a reproducible research-oriented pipeline for public benchmark evaluation, lightweight models, and TensorFlow Lite export.

## Motivation

- Wearable and edge AI for human-centered sensing.
- Real-time activity recognition from low-dimensional motion signals.
- Low-resource inference suitable for mobile or embedded deployment.
- Personalized/user-defined action recognition beyond fixed activity labels.
- Clear evaluation protocols that separate public benchmark results from small self-collected demos.

## Current Status

- The original Python prototype is still present in the root scripts (`recordData.py`, `processData.py`, `model.py`, `main.py`, `visualization.py`).
- A new benchmark-oriented Python package has been added under `src/activity_recognition/`.
- WISDM integration, subject-wise splitting, MLP/1D-CNN baselines, training/evaluation scripts, TensorFlow Lite export, and local CPU latency benchmarking are implemented.
- The Android data-collection client exists separately at <https://github.com/codemaivanngu/CollectAccelerometerDatav2>.
- Full Android on-device inference is planned through TensorFlow Lite, but this repository currently benchmarks TFLite on the developer machine CPU unless integrated into the Android client.

## Repository Relation

The Android data collection and streaming client is maintained separately:

<https://github.com/codemaivanngu/CollectAccelerometerDatav2>

Current prototype architecture:

```text
Android sensor client -> Python socket server/predictor -> real-time label/plot
```

Planned mobile architecture:

```text
Android sensor client -> on-device preprocessing -> TFLite inference -> real-time label display
```

Android Studio is only needed for the mobile demo. The Python benchmark pipeline in this repository can be run without Android Studio.

## Project Layout

```text
configs/                         WISDM experiment configs
src/activity_recognition/         New research-oriented Python package
scripts/                          CLI entry points for data, train, eval, export, benchmark
docs/                             Project protocol and limitation notes
results/                          Placeholder for lightweight result notes
tests/                            Smoke/unit checks for windowing and models
recordData.py                     Original socket data recorder
processData.py                    Original CSV preprocessing prototype
model.py                          Original Keras training experiment
main.py                           Original real-time Python predictor
visualization.py                  Original plotting helper
```

## Planned and Implemented Pipeline

Public benchmark path:

```text
WISDM dataset
-> preprocessing/windowing
-> subject-wise train/validation/test split
-> train MLP or 1D-CNN
-> evaluate accuracy, macro-F1, classification report, confusion matrix
-> export TensorFlow Lite
-> benchmark local CPU latency and model size
```

Personalized demo path:

```text
self-collected accelerometer data
-> user-defined labels
-> train or fine-tune lightweight classifier
-> real-time prediction prototype
```

Self-collected data is treated as a personalization and user-defined action demo, not as standalone scientific evidence of broad generalization.

## Install

Preferred setup with `uv`:

```bash
uv sync
```

Fallback setup with `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local imports from `src/`, run scripts from the repository root through `uv run` or an activated environment. For test runners, `pytest.ini` configures `src` as the Python path.

## Prepare WISDM

The repository does not commit WISDM data. Use the helper script:

```bash
uv run python scripts/download_wisdm.py --out data/raw/wisdm
```

If automatic download fails, place the classic WISDM raw file here:

```text
data/raw/wisdm/WISDM_ar_v1.1_raw.txt
```

Expected raw row format:

```text
user,activity,timestamp,x,y,z;
```

## Train

Train the 1D-CNN baseline:

```bash
uv run python scripts/train_wisdm.py --config configs/wisdm_cnn1d.yaml
```

Train the MLP baseline:

```bash
uv run python scripts/train_wisdm.py --config configs/wisdm_mlp.yaml
```

Each run writes model and metadata under the configured `outputs/...` directory.

## Evaluate

```bash
uv run python scripts/evaluate_wisdm.py \
  --config configs/wisdm_cnn1d.yaml \
  --run-dir outputs/wisdm_cnn1d
```

Evaluation outputs include:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`

The main split is subject-wise to reduce leakage from overlapping windows across the same person.

## Export TensorFlow Lite

```bash
uv run python scripts/export_tflite.py \
  --model outputs/wisdm_cnn1d/model.keras \
  --out outputs/wisdm_cnn1d/model.tflite
```

Optional float16 weight quantization:

```bash
uv run python scripts/export_tflite.py \
  --model outputs/wisdm_cnn1d/model.keras \
  --out outputs/wisdm_cnn1d/model.float16.tflite \
  --float16
```

## Benchmark TFLite

```bash
uv run python scripts/benchmark_tflite.py \
  --model outputs/wisdm_cnn1d/model.tflite \
  --input-shape 1,128,4
```

This is a CPU benchmark on the developer machine, not a real Android latency measurement. Android latency should be measured inside the Android client or with Android benchmarking tools after integration.

## Evaluation Metrics

The benchmark pipeline reports:

- accuracy,
- macro-F1,
- per-class precision/recall/F1,
- classification report,
- confusion matrix,
- model size,
- TensorFlow Lite latency on local CPU.

## Original Prototype Workflow

The original user-defined action prototype is still available:

1. Run `recordData.py` to collect accelerometer data from the mobile client.
2. Run `processData.py` to create a training CSV from recorded data.
3. Run `model.py` to train a small Keras model.
4. Run `main.py` to start the socket predictor and visualization.

This workflow is useful for demonstrating personalization, but it should be evaluated separately from public benchmark results.

## Limitations

- The self-collected dataset is small and may reflect only a limited set of users, devices, and environments.
- Public benchmark evaluation is needed for fairer comparison and reproducibility.
- WISDM is a useful starting benchmark, but it does not cover all real-world personalized actions.
- The current Android integration is data collection/streaming oriented; complete on-device inference is still future work.
- Local CPU TFLite latency is not a substitute for measured Android latency.
- The project does not claim state-of-the-art performance.

## Roadmap

- WISDM benchmark experiments with subject-wise evaluation.
- Optional UCI HAR integration.
- TensorFlow Lite export and size/latency reporting.
- Android TFLite inference integration.
- Session-wise protocol for self-collected personalized actions.
- Few-shot or final-layer adaptation for user-defined actions.

## Smoke Check

Run a synthetic smoke test without downloading WISDM:

```bash
uv run python scripts/smoke_test.py
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

The environment variable prevents unrelated globally installed pytest plugins from being auto-loaded into this project environment.
