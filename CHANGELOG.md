# Changelog

All notable changes to this project will be documented in this file.

## 2026-03-15

### Added
- Speaker-independent split configuration in `configs/speaker_split_v1.json`.
- Shared dataset/split utilities in `src/utils/dataset.py`.
- Test speaker export file at `results/metrics/test_speakers.txt`.
- Project-level `README.md` documenting structure, workflow, and experiment rules.

### Changed
- Updated training pipeline in `src/models/train.py` to use fixed speaker-level train/val/test splits.
- Updated evaluation pipeline in `src/utils/save_results.py` to report on the fixed speaker-independent test split.
- Updated TFLite conversion sampling in `src/models/convert_tflite.py` to draw representative samples from the train split.
- Refreshed model artifacts and evaluation outputs after retraining.

### Research Notes
- Incomplete speaker recordings were preserved and used in training, not discarded.
- Validation/test now use unseen complete speakers only to prevent speaker leakage.
- Current speaker-independent test accuracy: **82.64%**.
