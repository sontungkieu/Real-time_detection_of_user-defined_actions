# Project Overview

This repository is being evolved from an early accelerometer prototype into a research-oriented activity recognition project.

The current positioning is:

**Personalized On-device Activity Recognition from Wearable-like Motion Signals**

The project has two complementary tracks:

- Public benchmark evaluation using WISDM and UCI HAR for reproducibility and scientific credibility.
- Self-collected accelerometer data for personalized/user-defined action recognition demos.

The new code lives under `src/activity_recognition/` and is exposed through scripts in `scripts/`. The original root scripts remain available as the legacy prototype workflow.

The project should not be described as state-of-the-art. It is a compact, reproducible engineering and research prototype focused on lightweight models, benchmark-specific evaluation protocols, TensorFlow Lite export, and future Android deployment.
