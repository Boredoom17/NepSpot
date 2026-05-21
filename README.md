
# NepSpot

Compact Nepali keyword-spotting research code and deployment artifacts. NepSpot extracts MFCC features from short clips, trains compact KWS models (Vanilla CNN, DS-CNN, BC-ResNet), and provides model export paths for edge deployment (TFLite / Arduino Nano 33 BLE Sense).

Goals
- Reproducible experiments with speaker-independent splits and seeded training.
- Small-footprint models suitable for microcontrollers.
- Clear data-mining and provenance tracking for external audio sources.

Quick Start
1. Activate the project's venv and install dependencies (see `requirements.txt`).
2. Place audio under `data/raw/` or use `data/external/` staging for incoming corpora.
3. Run augmentation: `python3 src/utils/run_augmentation.py`.
4. Extract MFCCs: `python3 src/features/extract_mfcc.py`.
5. Train models: see `src/models/train_vanilla_phase1.py`, `src/models/train_phase1.py`, `src/models/train_bcresnet_phase1.py`.
6. Convert/export to TFLite: training scripts export SavedModel and call PTQ conversion for INT8.

Reproducibility
- All Phase-1 training drivers read `SEED` from the environment (default 42) and set `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and `tf.random.set_seed` to ensure deterministic runs where feasible.
- Feature extraction and normalization stats are saved under `models/saved/` for repeatable preprocessing.

Repository Layout (short)
- `configs/` — experiment configs and speaker splits
- `data/raw/`, `data/processed/` — audio and extracted MFCCs
- `models/saved/`, `models/tflite/` — trained artifacts and TFLite
- `results/` — metrics and figures
- `src/` — project source: `features/`, `models/`, `utils/`, `inference/`
- `scripts/` — helpers: dataset building, audits, and analysis

Documentation & Context
- See `CONTEXT.md` for an annotated snapshot of the project state and measured on-device numbers.
- See `CHANGELOG.md` for a dated history of major changes and dataset/experiment notes.

Citation
If you use this code in research, please cite the repository and contact the author for the preferred bibliographic entry.

License
This repository is published under the MIT License (see LICENSE file).

Questions or contributions: open an issue or PR.
