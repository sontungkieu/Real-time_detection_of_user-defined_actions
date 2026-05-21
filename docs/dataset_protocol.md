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

## Metrics

Evaluation reports:

- accuracy,
- macro-F1,
- weighted-F1,
- classification report,
- confusion matrix.

Model size and local CPU TFLite latency are reported after export.
