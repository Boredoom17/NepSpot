
# NepSpot (Work in Progress)

Hi! this is NepSpot — my attempt to bring Nepali voice commands (for now on low scale) to low-cost devices, fully offline, aiming to make keyword spotting in Nepali accessible for everyone, even on tiny hardware like the Arduino Nano 33 BLE Sense.

NepSpot collects real Nepali voice commands, extracts MFCC features, trains a compact DS-CNN model, and runs everything on-device — no internet needed. The goal: make Nepali speech tech practical for local, resource-constrained settings.


## Research Setup

- Validation and test splits are speaker-independent, so the model is tested on voices it’s never heard before.
- Any incomplete or partial recordings are used for training only.
- The split configuration is in `configs/speaker_split_v1.json`.
- Dataset loading and helpers are in `src/utils/dataset.py`.

This way, I can use all the data I collected (even the messy bits) for training, but keep the evaluation fair and honest for the paper.


## Project Layout

- `configs/` — experiment configs and speaker splits
- `data/raw/` — all the original and augmented audio clips
- `data/processed/` — MFCC features, organized by speaker and keyword
- `models/saved/` — trained Keras models, label classes, stats
- `models/tflite/` — TFLite models (float32 and INT8) for deployment
- `results/figures/` — plots and visualizations
- `results/metrics/` — evaluation reports and metrics
- `src/features/` — feature extraction scripts
- `src/models/` — model code and TFLite conversion
- `src/inference/` — live mic and test scripts
- `src/utils/` — helpers for data, augmentation, and results


## How to Reproduce (or Play With) NepSpot

1. Put your recordings in `data/raw/` (or use mine if you have access).
2. Run `src/utils/run_augmentation.py` to boost the dataset for under-represented words.
3. Extract MFCC features with `src/features/extract_mfcc.py`.
4. Train the model using `src/models/train.py`.
5. Evaluate results with `src/utils/save_results.py`.
6. Export TFLite models for Arduino with `src/models/convert_tflite.py`.
7. Try live keyword spotting on your laptop with `src/inference/live_mic.py` or test individual words with `src/inference/test_words.py`.


## Notes & Tips

- If you add new audio or augmentations, don’t forget to re-extract MFCCs before retraining.
- Always keep the test speakers the same for fair comparisons.
- For the main results, only use the fixed speaker-independent split.
- If you want to experiment with partial speakers, just change the train set—leave validation and test untouched.

---

If you have questions, want to collaborate, or just want to chat about Nepali speech tech, feel free to reach out!
