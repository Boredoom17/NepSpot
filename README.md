# NepSpot (Work in Progress)

NepSpot is an offline Nepali keyword spotting project for low-resource IoT and TinyML use cases. The current pipeline collects Nepali voice commands, extracts MFCC features, trains a DS-CNN model, evaluates it with speaker-independent splits, and exports deployable TFLite models for embedded hardware such as Arduino Nano 33 BLE Sense.

## Research Setup

- Validation and test are speaker-independent.
- All incomplete or partially recorded speakers are preserved in training only.
- The fixed split lives in `configs/speaker_split_v1.json`.
- Shared dataset loading utilities live in `src/utils/dataset.py`.

This keeps personal and low-volume recordings in the project while protecting the main paper metrics from leakage.

## Project Layout

- `configs/`
  Fixed experiment configuration, including the speaker split.
- `data/raw/`
  Original and augmented `.wav` files.
- `data/processed/`
  Normalized MFCC `.npy` features grouped by speaker and keyword.
- `models/saved/`
  Trained Keras model, label classes, MFCC stats, and SavedModel export.
- `models/tflite/`
  Float32 and INT8 deployment artifacts.
- `results/figures/`
  Evaluation plots.
- `results/metrics/`
  Classification reports and split-specific evaluation outputs.
- `src/features/`
  Feature extraction scripts.
- `src/models/`
  Training, architecture, and TFLite conversion.
- `src/inference/`
  Live microphone and interactive testing scripts.
- `src/utils/`
  Dataset helpers, augmentation, results export, and data download helpers.

## Recommended Workflow

1. Download or collect recordings into `data/raw/`.
2. Augment under-recorded words with `src/utils/run_augmentation.py`.
3. Rebuild normalized MFCC features with `src/features/extract_mfcc.py`.
4. Train with `src/models/train.py`.
5. Evaluate with `src/utils/save_results.py`.
6. Export embedded models with `src/models/convert_tflite.py`.
7. Run laptop-side inference checks with `src/inference/live_mic.py` or `src/inference/test_words.py`.

## Important Notes

- If you add new recordings or new augmentations, rerun MFCC extraction before training.
- Do not change the test speakers between model comparisons.
- For the paper's main table, report only the fixed speaker-independent split.
- If you want an ablation using partial speakers, keep the same validation and test sets and only change the train set.