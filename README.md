# NepSpot

NepSpot is a Nepali keyword-spotting system designed to run on a 256 KB-RAM Arduino Nano 33 BLE Sense (nRF52840). The repository contains the training and evaluation code, the INT8-quantized TFLite model weights, and the on-device firmware used in the accompanying paper.

The deployed BC-ResNet model recognizes 12 Nepali command words plus `_silence` and `_unknown` from 16 kHz audio captured by the board's PDM microphone. MFCC feature extraction and inference both run on-device with no network dependency.

## Keywords

`aghillo`, `arko`, `baalnu`, `banda`, `feri`, `hoina`, `huncha`, `maathi`, `roknu`, `suru`, `tala`, `thik_chha`, plus the auxiliary classes `_silence` and `_unknown`.

## Repository layout

```
configs/                 Experiment configs (speaker splits, k-fold splits, mining sources)
data/                    Pointers and manifests only — audio held out of repo (see data/README.md)
firmware/NepSpot/        Arduino sketch + INT8 model header (open NepSpot.ino in Arduino IDE)
models/saved/v6/         Canonical Keras/SavedModel checkpoints (3 seeds x 3 architectures)
models/tflite/v6/        INT8 TFLite models matching v6 checkpoints
models/kfold/            5-fold speaker-independent checkpoints
results/figures/         Plots cited in the paper
results/metrics/         Numeric reports, bootstrap CIs, McNemar / Friedman tests, ablations
src/
  data/                  Dataset download, extraction, speaker-split builders
  features/              MFCC feature extraction
  augment/               SpecAugment, time-stretch, noise mixing
  models/                Architecture definitions: Vanilla CNN, DS-CNN, BC-ResNet
  training/              Training drivers and shared training utilities
  evaluation/            Bootstrap CIs, McNemar, ROC/FRR/FAR, ensemble, SVM baseline, OOV
  conversion/            Post-training quantization, TFLite-to-C-header export
  inference/             Live microphone demo and offline test-clip runner
notebooks/               Audio exploration notebook
requirements.txt
```

## Reproducing the paper results

The canonical numbers come from `src/training/run_seeds_experiment.py` driving Vanilla CNN, DS-CNN, and BC-ResNet on the speaker-independent split (`configs/speaker_split.json`) with seeds 42, 123, 456. Outputs land under `results/seeds_experiment_artifacts/` and `models/{saved,tflite}/v6/`.

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Features (requires raw audio under data/raw/, not distributed with this repo)
python3 src/features/extract_mfcc.py

# 3. Training — three seeds per architecture
SEED=42  python3 src/training/run_seeds_experiment.py
SEED=123 python3 src/training/run_seeds_experiment.py
SEED=456 python3 src/training/run_seeds_experiment.py

# 4. Statistical tests
python3 src/evaluation/bootstrap_ci.py
python3 src/evaluation/mcnemar_tests.py
python3 src/evaluation/ensemble.py

# 5. Speaker-independent 5-fold cross-validation (optional)
python3 src/training/build_kfold_splits.py
for f in 1 2 3 4 5; do python3 src/training/train_kfold_single.py --fold $f; done
```

All training drivers read `SEED` from the environment, set `PYTHONHASHSEED`, and seed `random`, NumPy, and TensorFlow for deterministic runs.

## On-device deployment

The deployed model is `firmware/NepSpot/model_int8.h` (BC-ResNet INT8, ~95 KB tensor data). Open `firmware/NepSpot/NepSpot.ino` in the Arduino IDE with the Arduino Nano 33 BLE Sense board package and `Chirale_TensorFlowLite` library installed, then upload at 115200 baud. The serial monitor at 115200 baud shows per-class scores after each detected utterance.

Measured on Arduino Nano 33 BLE Sense Rev2 (nRF52840 @ 64 MHz):

| Metric | Value |
|---|---|
| Inference latency (mean, n=8) | 659 ms |
| Flash usage | 500,168 / 983,040 bytes (50%) |
| SRAM (BSS) | 221,320 / 262,144 bytes (84%) |
| Free heap | 40,824 bytes |
| TFLM tensor arena | 130 KB |

## Data

Audio data is held privately and is not part of this repository. See `data/README.md`. Manifests describing source provenance (Common Voice clip IDs, OpenSLR utterances, internal recordings) are tracked under `data/external/manifests/`.

## Citation

A preprint describing NepSpot will be posted on arXiv. *arXiv link coming soon.*

```bibtex
@misc{nepspot,
  title  = {NepSpot: Nepali Keyword Spotting on Resource-Constrained Microcontrollers},
  author = {Chhetri, Aadarsha},
  year   = {2026},
  note   = {arXiv preprint, link forthcoming}
}
```

## License

MIT — see [LICENSE](LICENSE).
