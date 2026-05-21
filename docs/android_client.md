# Android Client

The related Android data-collection client is:

<https://github.com/codemaivanngu/CollectAccelerometerDatav2>

## Current Role

The Android client is used for accelerometer data collection and streaming. The original Python scripts in this repository receive sensor data through a socket server and perform real-time plotting/prediction.

Current architecture:

```text
Android sensor client -> Python server/predictor
```

## Planned Role

The intended deployment direction is:

```text
Android sensor client -> on-device preprocessing -> TFLite model inference -> real-time label display
```

The Python benchmark pipeline does not require Android Studio. Android Studio is only needed when running or modifying the mobile demo.

## Integration Notes

The TensorFlow Lite export scripts in this repository produce `.tflite` files that can later be copied into the Android project. Before claiming mobile latency, latency should be measured on the target Android device rather than inferred from the local CPU benchmark.
