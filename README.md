<div align="center">

# 🎙️ NepSpot

### Real-Time Nepali Keyword Spotting on a 256 KB Microcontroller

[![arXiv](https://img.shields.io/badge/arXiv-link%20coming%20soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](#citation)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE%20Sense-00979D?style=for-the-badge&logo=arduino&logoColor=white)](https://store.arduino.cc/products/arduino-nano-33-ble-sense)
[![C++](https://img.shields.io/badge/C%2B%2B-TFLite%20Micro-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://www.tensorflow.org/lite/microcontrollers)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)


</div>

---

## Overview

NepSpot is a keyword-spotting system for Nepali that recognizes 12 spoken command words on a microcontroller with **256 KB of RAM and 1 MB of flash** — no network, no phone, no cloud. The Arduino Nano 33 BLE Sense Rev2 captures audio through its onboard PDM microphone, computes MFCC features on-device, and runs an INT8-quantized BC-ResNet for inference. Everything fits in the device's tiny memory budget.

The repository contains the full training and evaluation pipeline (three compact architectures benchmarked under a speaker-independent split and 5-fold cross-validation), the INT8 TFLite model artifacts, and the Arduino firmware that was actually flashed and measured.

The 12 keywords:

`aghillo` (front) • `arko` (next) • `baalnu` (turn on) • `banda` (close) • `feri` (again) • `hoina` (no) • `huncha` (okay) • `maathi` (up) • `roknu` (stop) • `suru` (start) • `tala` (down) • `thik_chha` (correct)

## Key Features

- **On-device inference** — MFCC + BC-ResNet INT8 running locally; no network at any stage
- **Three architectures benchmarked** — Vanilla CNN, DS-CNN, BC-ResNet under identical conditions
- **Statistically rigorous evaluation** — 3-seed runs, speaker-resampled bootstrap CIs, McNemar and Friedman tests
- **5-fold speaker-independent cross-validation** for generalization estimates
- **Confidence-thresholded OOV rejection** with full FAR/FRR sweeps
- **Reproducible** — deterministic seeding, frozen config splits, archived per-seed reports

## Tech Stack

- **Training**: Python 3.10, TensorFlow 2.x, NumPy, SciPy, scikit-learn, librosa
- **Quantization**: TensorFlow Lite Converter (INT8 PTQ + QAT)
- **Inference (device)**: TensorFlow Lite Micro + ARM CMSIS-NN
- **Firmware**: C++ on Arduino Nano 33 BLE Sense Rev2 (nRF52840 @ 64 MHz)
- **Data collection**: Boli-Recorder web app + OpenSLR-54 mining

## Results

Test accuracy reported as **mean ± std** across 3 seeds (42, 123, 456) on the speaker-independent test split. Model sizes are INT8 TFLite flat-buffer sizes.

### Speaker-independent single split

| Model        | Accuracy            | Model size  |
|--------------|---------------------|-------------|
| Vanilla CNN  | **83.75% ± 1.12%**  | 122.55 KB   |
| DS-CNN       | 79.04% ± 1.15%      | **33.69 KB** |
| BC-ResNet    | 81.72% ± 2.20%      | 93.64 KB    |

### 5-fold speaker-independent cross-validation

| Model        | Accuracy            |
|--------------|---------------------|
| Vanilla CNN  | **75.15% ± 3.44%**  |
| DS-CNN       | 62.93% ± 2.95%      |
| BC-ResNet    | 67.19% ± 6.41%      |

### On-device deployment (BC-ResNet INT8 on Arduino Nano 33 BLE Sense Rev2)

| Metric                | Value                         |
|-----------------------|-------------------------------|
| Inference latency     | **659 ms** (n=8)              |
| Flash usage           | 500,168 / 983,040 B (50%)     |
| SRAM (BSS)            | 221,320 / 262,144 B (84%)     |
| Free heap             | 40,824 B                      |
| TFLM tensor arena     | 130 KB                        |

## How to Reproduce

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Extract MFCC features (requires raw audio under data/raw/)
python3 src/features/extract_mfcc.py

# 3. Train each architecture with three seeds
SEED=42  python3 src/training/run_seeds_experiment.py
SEED=123 python3 src/training/run_seeds_experiment.py
SEED=456 python3 src/training/run_seeds_experiment.py

# 4. Statistical tests
python3 src/evaluation/bootstrap_ci.py
python3 src/evaluation/mcnemar_tests.py
python3 src/evaluation/ensemble.py

# 5. 5-fold cross-validation (optional)
python3 src/training/build_kfold_splits.py
for f in 1 2 3 4 5; do
    python3 src/training/train_kfold_single.py --fold $f
done
```

### Deploying to Arduino

1. Open `firmware/NepSpot/NepSpot.ino` in the Arduino IDE.
2. Install the board package **Arduino Mbed OS Nano Boards** and the library **Chirale_TensorFlowLite**.
3. Select board: *Arduino Nano 33 BLE Sense*; upload.
4. Open Serial Monitor at **115200 baud** to watch per-class scores after each detected utterance.

## Dataset

The training corpus was assembled from two sources, both held privately:

- **30 native Nepali speakers** recorded via the **Boli-Recorder** web app (12 keywords × 10 takes per speaker).
- **443 speakers** mined from **OpenSLR-54** (Nepali ASR corpus) for additional acoustic diversity, augmented with SpecAugment and Mixup.

Audio data is not shipped in this repository. See [data/README.md](data/README.md). Manifests describing source provenance live under [data/external/manifests/](data/external/manifests/).

## Project Structure

```
NepSpot/
├── configs/                  # Speaker splits, k-fold splits, data-mining configs
├── data/                     # Manifests + provenance only (audio held privately)
├── firmware/NepSpot/         # Arduino sketch + INT8 model header
│   ├── NepSpot.ino
│   └── model_int8.h
├── models/
│   ├── saved/v6/             # Keras + SavedModel checkpoints (3 seeds × 3 archs)
│   ├── tflite/v6/            # INT8 TFLite models
│   └── kfold/                # Per-fold INT8 models
├── results/
│   ├── figures/              # ROC curves, confusion matrices, per-keyword accuracy
│   ├── metrics/              # Reports, bootstrap CIs, McNemar / Friedman tests
│   ├── ablation/             # SpecAugment + Mixup ablation
│   └── seeds_experiment_artifacts/
├── src/
│   ├── data/                 # Dataset loading, downloading, speaker splits
│   ├── features/             # MFCC extraction
│   ├── augment/              # SpecAugment, time-stretch, noise
│   ├── models/               # Vanilla CNN, DS-CNN, BC-ResNet definitions
│   ├── training/             # Training drivers and shared utilities
│   ├── evaluation/           # Bootstrap, McNemar, ensemble, SVM baseline, OOV
│   ├── conversion/           # PTQ → INT8 TFLite → C header
│   └── inference/            # Live microphone demo + offline test runner
├── notebooks/                # Audio exploration
└── requirements.txt
```

## Citation

```bibtex
@misc{chhetri2026nepspot,
  title  = {NepSpot: Real-Time Nepali Keyword Spotting on a 256 KB Microcontroller},
  author = {Chhetri, Aadarsha},
  year   = {2026},
  note   = {arXiv preprint, link forthcoming}
}
```

*[arXiv link coming soon]*

## Acknowledgments

Thanks to every speaker who contributed recordings through Boli-Recorder, and to the OpenSLR-54 contributors whose corpus made the augmentation pipeline possible. <br>

---

<div align="center">

**A hope of small contribution to Nepali NLP.**

</div>
