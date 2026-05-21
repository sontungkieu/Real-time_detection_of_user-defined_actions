# Dataset Protocol

## Public Benchmark

The first supported public benchmark is WISDM. Raw data is expected under:

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
