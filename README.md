
# NepSpot (Work in Progress)

Hi! this is NepSpot — my attempt to bring Nepali voice commands (for now on low scale) to low-cost devices, fully offline, aiming to make keyword spotting in Nepali accessible for everyone, even on tiny hardware like the Arduino Nano 33 BLE Sense.

NepSpot collects real Nepali voice commands, extracts MFCC features, trains compact keyword spotting models, and runs everything on-device — no internet needed. The goal: make Nepali speech tech practical for local, resource-constrained settings.

Current model baselines in this repo:
- DS-CNN baseline (deployment-friendly, smaller footprint)
- Vanilla CNN baseline (higher capacity reference)
- BC-ResNet-1 baseline (Broadcasted Residual Learning style architecture for efficient KWS)


## Research Setup

- Validation and test splits are speaker-independent, so the model is tested on voices it’s never heard before.
- Any incomplete or partial recordings are used for training only.
- The split configuration is in `configs/speaker_split_v1.json`.
- Dataset loading and helpers are in `src/utils/dataset.py`.

This way, I can use all the data I collected (even the messy bits) for training, but keep the evaluation fair and honest for the paper.


## Project Layout

- `configs/` — experiment configs and speaker splits
- `configs/data_mining/` — keyword aliases and source planning configs for external mining
- `data/raw/` — all the original and augmented audio clips
- `data/processed/` — MFCC features, organized by speaker and keyword
- `data/external/` — pre-mining workspace (incoming source dumps, manifests, review logs, staging)
- `models/saved/` — trained Keras models, label classes, stats
- `models/tflite/` — TFLite models (float32 and INT8) for deployment
- `results/figures/` — plots and visualizations
- `results/metrics/` — evaluation reports and metrics
- `src/features/` — feature extraction scripts
- `src/models/` — model code and TFLite conversion
- `src/inference/` — live mic and test scripts
- `src/utils/` — helpers for data, augmentation, and results
- `scripts/data_mining/` — helper scripts for creating and maintaining the mining scaffold
- `docs/dataset-expansion/` — beginner-friendly Option 4 workflow docs


## Dataset Expansion Workspace (Option 4)

Before mining internet data, initialize the workspace:

1. `bash scripts/data_mining/make_dataset_scaffold.sh`
2. Track candidate clips in `data/external/manifests/candidates.csv`.
3. Track accepted/rejected decisions in `data/external/manifests/accepted.csv` and `data/external/manifests/rejected.csv`.
4. Keep source provenance in `data/external/manifests/source_registry.csv`.
5. Follow the beginner guide at `docs/dataset-expansion/step-by-step-beginner.md`.

This keeps external mining clean and separate from model-ready `data/raw/` and `data/processed/`.


## How to Reproduce (or Play With) NepSpot

1. Put your recordings in `data/raw/` (or use mine if you have access).
2. Run `src/utils/run_augmentation.py` to boost the dataset for under-represented words.
3. Extract MFCC features with `src/features/extract_mfcc.py`.
4. Train DS-CNN baseline with `src/models/train.py`.
5. Train Vanilla CNN baseline with `src/models/train_vanilla.py`.
6. Train BC-ResNet baseline with `src/models/train_bcresnet.py`.
7. Evaluate legacy DS-CNN results with `src/utils/save_results.py`.
8. Export generic TFLite models with `src/models/convert_tflite.py`, or use `src/models/train_vanilla.py` / `src/models/train_bcresnet.py` to train + quantize + report in one run.
9. Try live keyword spotting on your laptop with `src/inference/live_mic.py` or test individual words with `src/inference/test_words.py`.

## BC-ResNet Notes

- BC-ResNet architecture file: `src/models/bc_resnet.py`
- End-to-end BC-ResNet pipeline: `src/models/train_bcresnet.py`
- Expected outputs from BC-ResNet training run:
	- `models/saved/bc_resnet_best.keras`
	- `models/saved/bc_resnet_saved_model/`
	- `models/tflite/bc_resnet_int8.tflite`
	- `models/tflite/bc_resnet_int8.h`
	- `results/metrics/bc_resnet_report.txt`
- The BC-ResNet training script also prints a 3-way comparison table (Vanilla CNN, DS-CNN, BC-ResNet) when reference artifacts are present.


## Notes & Tips

- If you add new audio or augmentations, don’t forget to re-extract MFCCs before retraining.
- Always keep the test speakers the same for fair comparisons.
- For the main results, only use the fixed speaker-independent split.
- If you want to experiment with partial speakers, just change the train set—leave validation and test untouched.

---

If you have questions, want to collaborate, or just want to chat about Nepali speech tech, feel free to reach out!
