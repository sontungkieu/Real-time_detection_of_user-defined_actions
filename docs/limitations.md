# Limitations

- The original self-collected data is small and may be specific to one device, user, and environment.
- WISDM is a useful public benchmark but is not a complete proxy for personalized real-world behavior.
- UCI HAR adds a second public benchmark, but its official train/test split and pre-windowed inertial signals are not directly comparable to the WISDM subject-wise sliding-window protocol.
- Subject-wise evaluation reduces leakage but may produce class imbalance depending on subject/activity coverage.
- Repeated WISDM seeds show meaningful split variance. Current 1D-CNN accuracy across seeds 42/43/44 is 0.818 +/- 0.066, and macro-F1 is 0.747 +/- 0.090.
- Stair activities are the weakest area in the current WISDM runs. `Upstairs` and `Downstairs` are often confused with each other or with `Walking`, while `Jogging`, `Sitting`, and `Standing` are much stronger.
- Current UCI HAR baselines are first-pass sanity checks, not tuned benchmark submissions. The largest observed confusions are between `SITTING`/`STANDING` and between nearby walking/stair classes.
- Local CPU TensorFlow Lite benchmarking is not equivalent to Android device latency.
- Current Android integration is primarily data collection and streaming. Full on-device preprocessing and inference still require Android-side implementation.
- The repository should not claim state-of-the-art performance.
