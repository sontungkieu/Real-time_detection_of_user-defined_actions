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

If the Fordham endpoint times out, the same archive can also be fetched from the Google Drive mirror referenced in the Curiousily WISDM tutorial:

```bash
uvx gdown 152sWECukjvLerrVG2NUO8gtMFg83RKCF -O data/raw/wisdm/WISDM_ar_latest.tar.gz
tar -xzf data/raw/wisdm/WISDM_ar_latest.tar.gz -C data/raw/wisdm
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

The benchmark pipeline reports both classification quality and deployment-oriented efficiency metrics.

| Metric | What it measures | Why it matters here |
| --- | --- | --- |
| Accuracy | Fraction of test windows whose predicted class equals the ground-truth class. | Useful as a simple headline score, but can hide poor minority-class performance. |
| Precision per class | Of the windows predicted as a class, the fraction that are actually that class. | Helps identify false positives, for example predicting `jogging` too often. |
| Recall per class | Of the true windows from a class, the fraction recovered by the model. | Helps identify missed activities, for example failing to detect `walking` windows. |
| F1 per class | Harmonic mean of per-class precision and recall. | Balances false positives and false negatives for each activity. |
| Macro-F1 | Unweighted mean of per-class F1 scores. | Treats every activity equally, which is important when classes are imbalanced. |
| Weighted-F1 | Mean of per-class F1 weighted by test support. | Summarizes performance while accounting for class frequency. |
| Support | Number of test windows for each class. | Gives context for whether a per-class score is based on enough examples. |
| Classification report | Text/table summary of precision, recall, F1, and support. | Makes per-class behavior easy to inspect. |
| Confusion matrix | Counts of true labels versus predicted labels. | Shows which activities are confused with each other. |
| Model parameters | Number of trainable/non-trainable Keras parameters. | Rough proxy for model complexity and memory needs. |
| Keras model size | Size of the saved `.keras` model file. | Useful for artifact storage and comparing model variants before export. |
| TFLite model size | Size of the exported `.tflite` file. | Directly relevant to mobile/on-device deployment. |
| Mean latency | Average TFLite inference time across benchmark runs. | Quick estimate of typical local CPU inference cost. |
| Median latency | Middle TFLite inference time across benchmark runs. | More robust than mean when a few runs are slow. |
| P95 latency | 95th percentile TFLite inference time. | Captures tail latency, which matters for real-time responsiveness. |

### Latest WISDM Run

The following numbers come from a real WISDM v1.1 raw accelerometer run on this development machine. The original Fordham endpoint timed out from this environment, so the archive was downloaded from the Google Drive mirror referenced in the Curiousily WISDM tutorial and extracted to the ignored `data/raw/wisdm/` directory.

Run setup:

| Field | Value |
| --- | --- |
| Date | 2026-05-21 |
| Data source | WISDM v1.1 raw accelerometer file: `WISDM_ar_v1.1_raw.txt` |
| Parsed rows | 1,098,199 |
| Classes | `Downstairs`, `Jogging`, `Sitting`, `Standing`, `Upstairs`, `Walking` |
| Subjects | 36 |
| Split | Subject-wise; 25 train subjects, 5 validation subjects, 6 test subjects |
| Windows | 16,890 total; 11,528 train, 2,476 validation, 2,886 test |
| Window shape | `128 x 4` (`x`, `y`, `z`, magnitude) |
| Model | 1D-CNN |
| Epochs | 30 |
| Final train accuracy / validation accuracy | 0.986 / 0.686 |
| Best validation accuracy | 0.791 at epoch 1 |
| Python / TensorFlow | Python 3.11.6 / TensorFlow 2.21.0 |
| Local CPU | Intel Core i5-9300H CPU @ 2.40GHz |

Classification results:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.787 |
| Macro-F1 | 0.792 |
| Weighted-F1 | 0.801 |
| Test windows | 2,886 |
| `Downstairs` precision / recall / F1 / support | 0.380 / 0.505 / 0.434 / 323 |
| `Jogging` precision / recall / F1 / support | 0.996 / 0.941 / 0.968 / 828 |
| `Sitting` precision / recall / F1 / support | 0.985 / 1.000 / 0.992 / 65 |
| `Standing` precision / recall / F1 / support | 0.867 / 1.000 / 0.929 / 117 |
| `Upstairs` precision / recall / F1 / support | 0.520 / 0.656 / 0.580 / 445 |
| `Walking` precision / recall / F1 / support | 0.939 / 0.773 / 0.848 / 1,108 |

Deployment-oriented results:

| Metric | Result |
| --- | ---: |
| Model parameters | 11,430 |
| Keras model size | 0.164 MB |
| TFLite model size | 0.048 MB / 49.28 KB |
| TFLite benchmark input | `1,128,4` |
| TFLite benchmark runs | 500 measured + 50 warmup |
| Mean latency | 0.0231 ms |
| Median latency | 0.0220 ms |
| P95 latency | 0.0242 ms |

These latency numbers are local CPU measurements with TensorFlow Lite XNNPACK. They are not Android device latency.

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
