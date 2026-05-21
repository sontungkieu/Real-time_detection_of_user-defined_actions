# Dataset Protocol

## Public Benchmarks

The supported public benchmarks are WISDM and UCI HAR.

## WISDM

Raw WISDM data is expected under:

```text
data/raw/wisdm/
```

The classic raw WISDM format is:

```text
user,activity,timestamp,x,y,z;
```

The loader normalizes these fields to:

- `subject_id`
- `activity`
- `timestamp`
- `x`
- `y`
- `z`

## Windowing

The default configs use:

- window size: 128 samples,
- step size: 64 samples,
- features: `x`, `y`, `z`,
- optional magnitude feature: `sqrt(x^2 + y^2 + z^2)`.

Windows are created within each subject/activity group so that a window does not cross label or subject boundaries.

## Split Policy

The main split is subject-wise. Subjects are assigned to train, validation, or test sets with no overlap.

Normalization is fit on training windows only and then applied to validation/test windows. This avoids leakage from test-set statistics.

The train CLI accepts `--seed` and `--output-dir` overrides so repeated subject-wise split runs can be stored side by side. The helper `scripts/run_wisdm_seeds.py` runs train, evaluation, TensorFlow Lite export, and local CPU benchmarking for a list of split seeds.

## Checkpoint Policy

Default WISDM configs use:

- `ModelCheckpoint` on `val_accuracy`,
- `EarlyStopping` on `val_accuracy`,
- mode: `max`,
- patience: 5,
- restored best validation weights.

For runs with validation data, `outputs/.../model.keras` is saved from the best validation checkpoint rather than the final epoch. `outputs/.../best_model.keras` is kept as the raw checkpoint artifact.

## Metrics

Evaluation reports:

- accuracy,
- macro-F1,
- weighted-F1,
- classification report,
- confusion matrix,
- raw confusion matrix JSON,
- top off-diagonal confusion pairs as JSON and text.

Model size and local CPU TFLite latency are reported after export.

Current WISDM confusion analysis shows that `Jogging`, `Sitting`, and `Standing` are strongest, while `Upstairs` and `Downstairs` are harder and are commonly confused with each other or with `Walking`.

## UCI HAR

UCI HAR data is expected under:

```text
data/raw/uci_har/UCI HAR Dataset/
```

Download helper:

```bash
uv run python scripts/download_uci_har.py --out data/raw/uci_har
```

UCI HAR already provides pre-windowed 128-step smartphone inertial signals, so the loader does not apply WISDM-style sliding windows. The first configs use six inertial channels:

- `total_acc_x`
- `total_acc_y`
- `total_acc_z`
- `body_gyro_x`
- `body_gyro_y`
- `body_gyro_z`

The first UCI HAR protocol uses the official train/test split. Validation is split from official training subjects for checkpoint selection. This is not identical to the WISDM subject-wise train/validation/test protocol and should be described separately in reports.

UCI HAR evaluation reports the same core metrics and artifacts as WISDM:

- accuracy,
- macro-F1,
- weighted-F1,
- classification report,
- confusion matrix,
- raw confusion matrix JSON,
- top off-diagonal confusion pairs as JSON and text.
