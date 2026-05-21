# Changelog

All notable changes to this project will be documented in this file.

## 2026-05-18

### Added
- `nepspot_context.md` — new section "2026-05-18: BC-ResNet Arduino
  deployment — measured (work done 2026-05-17)" with the headline
  on-device numbers that the paper will cite for edge inference.

### On-device deployment landed (BC-ResNet INT8, new architecture)
- **Latency:** **659 ms** (mean of n=8 runs, std < 1 ms) on Arduino
  Nano 33 BLE Sense Rev2 (nRF52840 @ 64 MHz).
- **Flash usage:** 500,168 / 983,040 bytes (50%).
- **SRAM (BSS):** 221,320 / 262,144 bytes (84%).
- **Free heap:** 40,824 bytes.
- **TFLM tensor arena:** 130 KB.
- **Op resolver:** 26 ops — added `AddLogistic` to support the
  Swish-via-`Logistic`+`Mul` decomposition emitted by the TFLite
  converter for the time-branch Swish activations.

### Changed
- The 2026-05-12 entry stating *"Arduino latency for BC-ResNet is
  331 ms"* is **superseded for paper purposes**. That measurement
  was on the old broken freq-only architecture. The paper's
  on-device numbers come exclusively from the 2026-05-18 section in
  `nepspot_context.md`.
- The 2026-05-17 limitation bullet *"BC-ResNet has not been
  Arduino-deployed yet"* is resolved.

### Research Notes
- **Net trade vs DS-CNN firmware:** more flash (500 KB vs 343 KB,
  still well under the 983 KB budget) in exchange for **3× speedup**
  (659 ms vs ≈ 1,955 ms) and **2× more usable heap** (40,824 vs
  19,728 bytes). The SRAM headroom gain (84% vs 92% BSS) comes from
  BC-ResNet's smaller intermediate activations despite its greater
  architectural complexity.
- **Paper scope:** only BC-ResNet is deployed on-device. DS-CNN and
  Vanilla CNN on-device numbers will **not** be reported in the
  paper.

## 2026-05-17

### Added
- `nepspot_context.md` — new section "2026-05-17: Final paper-ready
  numbers" with the headline single-split N=3, 5-fold, ablation,
  ROC/FRR/FAR, model-size, and limitation tables that the paper will
  cite verbatim.

### Results landed (paper-ready, supersede all prior single-seed and
14-class numbers)
- **Single-split, 3 seeds, voicer28-30 (n=361 originals only):**
  - Vanilla CNN: **83.75% ± 1.12%** INT8 (macro F1 0.834)
  - DS-CNN:     **79.04% ± 1.15%** INT8 (macro F1 0.785)
  - BC-ResNet:  **81.72% ± 2.20%** INT8 (macro F1 0.820)
  - Sources: `results/seeds_experiment_artifacts/{vanilla,dscnn,bcresnet}_seed{42,123,456}_report.txt`
- **5-fold cross-validation, 20-voicer pool
  (voicer1-17, 20, 22, 24; voicer18/19/21/23 missing on disk;
  voicer25-30 sacred held-out):**
  - Vanilla CNN: **75.15% ± 3.44%**
  - DS-CNN:     **62.93% ± 2.95%**
  - BC-ResNet:  **67.19% ± 6.41%**
  - Sources: `results/kfold/{vanilla,dscnn,bcresnet}_fold{1..5}_report.txt`
- **Ablation, DS-CNN seed 42:** no SpecAug + no Mixup = **79.50%**
  INT8 (vs 79.04% ± 1.15% with both). Within seed-noise band — the
  augmentation recipe contributes no measurable signal on top of the
  upstream wav-level augmentation.
  Source: `results/metrics/dscnn_no_specaug_report.txt`
- **ROC / FRR / FAR (BC-ResNet INT8 seed 42):** Macro AUC **0.974**,
  Macro EER **7.22%**, FRR @ FAR=1% = **21.07%**. Best per-keyword:
  `feri`, `huncha`, `maathi` (AUC > 0.998). Worst: `tala` (AUC 0.829).
  Source: `results/metrics/frr_far_roc_new.txt`

### Changed
- Earlier Phase 1 single-seed numbers (DS-CNN 84.86%, BC-ResNet
  86.06% / 80.33%, Vanilla "92.24%") and **all 14-class numbers** are
  marked **superseded for paper purposes** by the 2026-05-17 section
  in `nepspot_context.md`. They remain on disk as historical record.

### Research Notes
- **Single-split was ~10 pp optimistic vs k-fold** across every model
  (Vanilla 83.8 → 75.2, DS-CNN 79.0 → 62.9, BC-ResNet 81.7 → 67.2).
  voicer28-30 are easy speakers. Any single-split-only result here
  should be assumed to overstate generalisation.
- **Vanilla CNN wins both protocols** but is not Arduino-deployable
  (114,956 params, 122.55 KB INT8) — it remains an upper-bound
  reference rather than a candidate model.
- **BC-ResNet beats DS-CNN on k-fold** (67.19% vs 62.93%) with fewer
  parameters (13,732 vs 17,164). It also has the **highest
  fold-to-fold variance** (± 6.41%), so the paper should report
  per-fold numbers, not just the mean.
- **BC-ResNet seed 42 in the N=3 result is the h5-weight-transferred
  recovered model**, not a fresh in-train QAT→INT8 run. Seeds 123 and
  456 are fresh. A fresh seed-42 retrain would clean this up but was
  deferred — flagged in the corresponding section of
  `nepspot_context.md`.

## 2026-05-16

### Added
- `scripts/run_seeds_experiment.py` — subprocess wrapper that runs each of
  BC-ResNet / Vanilla CNN / DS-CNN with seeds [42, 123, 456] (9 runs total) and
  writes `results/metrics/seeds_experiment_report.txt` (per-run table +
  per-model mean ± std) and `results/metrics/seeds_experiment_raw.json`. Per-run
  artifacts archived under `results/seeds_experiment_artifacts/` to prevent
  the training scripts' fixed output paths from clobbering prior runs.
- `scripts/train_dscnn_no_specaug.py` — single-seed (42) DS-CNN ablation that
  replaces the SpecAug+Mixup pipeline with a plain
  `from_tensor_slices → shuffle → batch → prefetch` to quantify the
  augmentation contribution against the 79.04% ± 1.15% INT8 baseline.
- `scripts/recover_bcresnet_int8.py` (carried from 2026-05-15) — kept on disk
  as a known-good fallback now that the canonical in-train pipeline is fixed.
- ROC-related outputs for the new BC-ResNet INT8:
  - `results/metrics/frr_far_roc_new.txt` (per-keyword AUC, EER, FRR @ FAR ∈
    {0.1%, 1%, 5%}, macro averages)
  - `results/figures/roc_curves_new.png` (12 per-keyword ROC + macro-ROC)
- Canonical BC-ResNet seed-42 artifacts produced by the corrected in-train
  pipeline:
  - `models/tflite/bc_resnet_int8_phase1.tflite` (93.64 KB; INT8 80.33% /
    macro F1 0.8054)
  - `results/metrics/bc_resnet_phase1_report.txt`

### Changed
- `compute_roc_frr_far.py` rewritten: removed a double-normalisation bug
  (inputs from `data/processed/` are already mean/std-normalised by
  `src/features/extract_mfcc.py:91`); added FRR @ fixed-FAR operating-point
  reporting; interpolated EER instead of nearest-index; dequantised INT8
  output logits so reported thresholds are probability-scale; output paths
  changed to `frr_far_roc_new.{txt,png}`.
- All three Phase 1 training scripts and `phase1_training_utils.py` now read
  the random seed from the `SEED` env var (default 42), threading it through
  `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, `tf.random.set_seed`,
  the SpecAugment / Mixup `_BASE_SEED`, and the tf.data shuffle seed. No
  recipe changes — only seed plumbing.

### Research Notes
- The 2026-05-15 BC-ResNet INT8 number (84.76% / 0.8474) is from a
  weight-transferred recovery model and is retained on disk
  (`bc_resnet_int8_recovered.tflite`) as evidence the h5 recovery approach
  is numerically sound. The forward-looking canonical BC-ResNet INT8 number
  is 80.33% / 0.8054 from the seed-42 in-train QAT→INT8 path.
- Macro AUC of the new BC-ResNet INT8 is 0.9738 ± 0.0484 across the 12
  keywords. `tala` remains the laggard (AUC 0.829, EER 26.67%); `feri`,
  `huncha`, `maathi` reach AUC ≥ 0.999 with FRR@1%FAR = 0%.
- N=3 seed experiment infrastructure is verified ready (syntax, parser,
  table rendering all checked) but the 9-run experiment has not been
  executed yet; running it is the highest-priority remaining task before
  publishable comparison numbers can be claimed.

## 2026-05-15

### Audit + architecture overhaul
- BC-ResNet `src/models/bc_resnet.py` rewritten to the correct Kim 2021
  topology: frequency branch DW(3,1) + time branch DW(1,3) with dilation
  along time, broadcast via a frozen DepthwiseConv2D (kernel (2F-1, 1),
  weights 1/F) that is mathematically equivalent to AvgPool+UpSampling but
  collapses to a single quantisation scale, fixing the
  `same scale constraint` error that previously blocked INT8 conversion.
  Old buggy file preserved at `src/models/bc_resnet_OLD_BROKEN.py`.
- Determinism prologue added to all four training scripts
  (`train_phase1.py`, `train_vanilla_phase1.py`, `train_bcresnet_phase1.py`,
  `run_kfold_crossval.py`): `TF_DETERMINISTIC_OPS=1`,
  `TF_CUDNN_DETERMINISTIC=1`, `enable_op_determinism()`,
  `tf.random.set_seed(42)`. `phase1_training_utils.py` rewritten for
  stateless aug ops, `num_parallel_calls=1`, `prefetch(1)`, seeded shuffle —
  verified bit-identical reruns on synthetic data.
- `scripts/recover_bcresnet_int8.py` added: opens the old `.keras` as a ZIP,
  reads `model.weights.h5`, walks `config.json` to map tfmot
  `QuantizeWrapperV2` entries to inner-layer weights, transfers by name to
  the new architecture, then PTQ-converts via `model.export → SavedModel`
  (working around a Keras-3 `from_keras_model` MLIR
  `missing attribute 'value'` error). Recovered INT8 = 84.76% / 0.8474.

## 2026-05-10

### Added
- BC-ResNet-1 model implementation in `src/models/bc_resnet.py`.
- Dedicated BC-ResNet training + evaluation + export pipeline in `src/models/train_bcresnet.py`.
- External data-mining scaffold under `data/external/` for incoming corpora, manifests, review logs, and staging.
- Data-mining helper script `scripts/data_mining/make_dataset_scaffold.sh`.
- Beginner-friendly Option 4 docs under `docs/dataset-expansion/`.
- Mining configuration templates under `configs/data_mining/`.
- BC-ResNet output/report paths:
	- `models/saved/bc_resnet_best.keras`
	- `models/saved/bc_resnet_saved_model/`
	- `models/tflite/bc_resnet_int8.tflite`
	- `models/tflite/bc_resnet_int8.h`
	- `results/metrics/bc_resnet_report.txt`

### Changed
- Project `README.md` now documents all three baselines (DS-CNN, Vanilla CNN, BC-ResNet).
- Reproduction steps in `README.md` now include `train_vanilla.py` and `train_bcresnet.py`.

### Research Notes
- BC-ResNet training follows the same speaker-independent split protocol and train/val/test handling as existing baselines.
- BC-ResNet INT8 conversion uses representative samples from the train split (`max_samples=200`) for apples-to-apples PTQ.
- The BC-ResNet training script prints a 3-way baseline comparison table when DS-CNN/Vanilla reference artifacts are available.

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
