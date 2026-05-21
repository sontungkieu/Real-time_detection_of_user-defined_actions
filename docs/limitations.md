# Limitations

- The original self-collected data is small and may be specific to one device, user, and environment.
- WISDM is a useful public benchmark but is not a complete proxy for personalized real-world behavior.
- Subject-wise evaluation reduces leakage but may produce class imbalance depending on subject/activity coverage.
- Local CPU TensorFlow Lite benchmarking is not equivalent to Android device latency.
- Current Android integration is primarily data collection and streaming. Full on-device preprocessing and inference still require Android-side implementation.
- The repository should not claim state-of-the-art performance.
