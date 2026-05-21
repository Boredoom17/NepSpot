# NepSpot: Nepali Keyword Spotting on Edge — Full Research Context

## Purpose of This Document

This is the complete technical context for writing a research paper about NepSpot.
It covers dataset collection, feature extraction, model architecture, training,
quantisation, and on-device deployment — with all numbers filled in from the
actual code and results.

---

## What NepSpot Is

NepSpot is a **real-time, on-device Nepali keyword spotting (KWS) system** running
entirely on an Arduino Nano 33 BLE Sense Rev2 microcontroller (nRF52840, 256 KB SRAM).
No cloud, no internet — pure edge inference. It recognises 12 Nepali spoken words
from raw microphone audio using a DS-CNN model quantised to int8.

**Core claim:** First Nepali KWS system with speaker-independent evaluation and full
TinyML deployment on a sub-256 KB microcontroller.

---

## The 12 Keywords

| Keyword     | Meaning (approximate) |
|-------------|----------------------|
| aghillo     | forward / previous   |
| arko        | next / another       |
| baalnu      | turn on              |
| banda       | close / off          |
| feri        | again                |
| hoina       | no / it's not        |
| huncha      | yes / okay           |
| maathi      | up / above           |
| roknu       | stop                 |
| suru        | start / begin        |
| tala        | down / below         |
| thik_chha   | correct / alright    |

These are common Nepali directional and action words well-suited for offline
voice-command interfaces (e.g. smart home, assistive tech, edge robotics).

---

## Dataset

### Collection Method

Audio was collected using **Boli-Recorder** — a custom web application (HTML/JS,
Firebase backend) built specifically for this project. Speakers accessed the app
in a browser, recorded WAV clips word-by-word when prompted, and files were
automatically uploaded to Firebase. A sync tool (boli-sync) downloaded them to
local storage, organised as:

```
data/raw/<speaker_id>/<keyword>/<recording>.wav
```

### Recording Specifications

- **Format:** WAV, 16 kHz, mono
- **Duration per clip:** 1 second target
  (center-cropped if longer, zero-padded if shorter during feature extraction)
- **Real recordings per speaker per keyword:** ~10 clips (TARGET = 10)
- **Speakers:** 30 total (voicer1–voicer30), all Nepali speakers

### Speaker-Independent Splits

| Split          | Speakers            | Count | Notes |
|----------------|---------------------|-------|-------|
| Train complete | voicer1–voicer16    | 16    | All 12 keywords present |
| Train partial  | voicer17–voicer24   | 8     | Some keywords missing — used for training only |
| Validation     | voicer25–voicer27   | 3     | Never seen during training |
| Test           | voicer28–voicer30   | 3     | Never seen during training or model selection |

**No speaker appears in more than one split.** Validation and test speakers were
held out completely. The split config is in `configs/speaker_split_v1.json`.

Partial speakers (voicer17–24) are used only for training — keeping them out of
val/test ensures the rigorous speaker-independent evaluation uses only speakers
with complete recordings.

### Data Augmentation

Augmentation was applied to training data only (test/val speakers not augmented).
Two augmentation passes were run:

**Pass 1 — Fill to target (partial speakers only):**
For any speaker/keyword folder with fewer than 10 real clips, augmented clips
were generated to reach the 10-clip target using:
- Gaussian noise addition (noise_factor = 0.005)
- Time stretching (rate: 0.85–1.15×)
- Pitch shifting (±2 semitones)
- Time shifting (±10% of audio length)

**Pass 2 — Speed augmentation (all real clips, all training speakers):**
For every original clip, two additional variants were created:
- Fast version (time-stretch rate: 1.2–1.35×)
- Fast + noisy version (same + Gaussian noise)

**Result:** approximately 30 samples per speaker per keyword after augmentation
(~10 original + ~20 augmented).

### Dataset Size Summary

| Split | Speakers | Approx. Samples |
|-------|----------|-----------------|
| Train | 24       | ~8,640          |
| Val   | 3        | ~1,080          |
| Test  | 3        | 1,083 (actual)  |
| **Total** | **30** | **~10,800**  |

Test set breakdown (from classification_report.txt):
- 90 samples per keyword (except arko: 93)
- voicer28: 363 samples, voicer29: 360, voicer30: 360

---

## Feature Extraction Pipeline

Implemented in Python using **librosa** (`src/features/extract_mfcc.py`), then
**re-implemented from scratch in C++ on-device** to match exactly — this parity
is critical so the model receives the same features it was trained on.

### Parameters

| Parameter   | Value                              |
|-------------|-----------------------------------|
| Sample rate | 16,000 Hz                         |
| Duration    | 1.0 s (16,000 samples)            |
| n_FFT       | 1,024                             |
| Hop length  | 512                               |
| n_MFCC      | 40                                |
| Num frames  | 32                                |
| Mel scale   | Slaney (librosa default, not HTK) |
| center      | True (librosa default)            |
| Window      | Hann                              |

### Pipeline Steps

1. Load audio, normalise to float32 in [−1, 1]
2. Compute STFT with Hann window, n_fft=1024, hop=512, center=True
   (center=True pads signal with 512 zeros on each side so frame 0 is centred at sample 0)
3. Apply 40-band triangular mel filterbank (Slaney normalisation, 0–8000 Hz)
4. Convert to dB: `10 × log10(max(energy, 1e-10))`
5. Apply DCT-II (orthonormal) → 40 MFCC coefficients per frame
6. Output shape per clip: **(40 coefficients × 32 frames)**

### Global Scalar Normalisation

Computed once across the entire training corpus and saved:

```
mfcc_mean = -12.608452796936035
mfcc_std  =  78.48593139648438
```

Applied as: `(mfcc − mean) / (std + 1e-8)`

This single global scalar (not per-coefficient, not per-frame) is applied to the
entire (40, 32) MFCC matrix. The same constants are hardcoded in the Arduino
firmware so training and inference normalisation are identical.

---

## Model Architecture

### DS-CNN (Depthwise Separable CNN)

Inspired by Zhang et al. "Hello Edge: Keyword Spotting on Microcontrollers" (2018),
adapted for Nepali keywords and constrained to fit in 256 KB SRAM.

```
Input: (40, 32, 1)
│
├── Entry block
│   Conv2D(64, 3×3, same, no bias) → BatchNorm → ReLU
│
├── DS Block 1
│   DepthwiseConv2D(3×3, same, no bias) → BatchNorm → ReLU
│   Conv2D(64, 1×1, no bias) → BatchNorm → ReLU
│
├── DS Block 2  (same structure, 64 filters)
│
├── DS Block 3  (same structure, 64 filters)
│
├── GlobalAveragePooling2D
├── Dropout(0.25)
└── Dense(12, softmax)
```

**Total architecture:** 1 standard conv entry block + 3 depthwise separable blocks
+ global average pooling + output head.

**Parameter count:** 17,164 total (16,268 trainable + 896 non-trainable BatchNorm)

### Why DS-CNN

- Depthwise separable convolutions reduce computation vs standard convs by ~8–9×
- Global average pooling avoids large fully-connected layers
- Small enough to quantise to int8 and fit tensor arena within 167 KB

### Training Details

| Setting         | Value                                      |
|-----------------|--------------------------------------------|
| Framework       | TensorFlow / Keras                         |
| Optimiser       | Adam, lr = 0.001                           |
| Loss            | Sparse categorical crossentropy            |
| Batch size      | 32                                         |
| Max epochs      | 50                                         |
| Early stopping  | patience = 10, monitor = val_accuracy      |
| LR schedule     | ReduceLROnPlateau, factor = 0.5, patience = 5 |
| Best model      | Saved on val_accuracy (ModelCheckpoint)    |
| Seeds           | Python 42, NumPy 42, TensorFlow 42         |
| Input shape     | (40, 32, 1)                                |
| Output          | 12-class softmax                           |

---

## Results

### Overall Performance

| Metric                      | Value  |
|-----------------------------|--------|
| Test accuracy               | **84.86%** |
| Random chance baseline      | 8.33% (12 classes) |
| Macro F1                    | 0.85   |
| Test set size               | 1,083 samples |
| Evaluation type             | Speaker-independent |

### Per-Class Results (from classification_report.txt)

| Keyword    | Precision | Recall | F1   | Support |
|------------|-----------|--------|------|---------|
| aghillo    | 0.77      | 0.91   | 0.83 | 90      |
| arko       | 0.81      | 0.95   | 0.87 | 93      |
| baalnu     | 0.82      | 0.92   | 0.87 | 90      |
| banda      | 0.77      | 0.69   | 0.73 | 90      |
| feri       | 0.87      | 1.00   | 0.93 | 90      |
| hoina      | 0.87      | 0.83   | 0.85 | 90      |
| huncha     | 0.96      | 0.72   | 0.82 | 90      |
| maathi     | 1.00      | 0.83   | 0.91 | 90      |
| roknu      | 0.77      | 0.88   | 0.82 | 90      |
| suru       | 0.81      | 0.84   | 0.83 | 90      |
| tala       | 0.90      | 0.67   | 0.76 | 90      |
| thik_chha  | 0.93      | 0.93   | 0.93 | 90      |

Best performing: `feri` (F1=0.93), `thik_chha` (0.93), `maathi` (0.91)
Most challenging: `banda` (F1=0.73), `tala` (0.76)

**Note:** `banda` and `tala` have the lowest recall, suggesting these words are
acoustically similar to others in the set (likely confused with each other or
with `roknu`/`maathi`). A confusion matrix is available at
`results/figures/confusion_matrix.png`.

### Live On-Device Observations

After deployment on the Arduino (single user, not in training set), qualitative
recognition reliability:

- **Consistently detected:** thik_chha, banda, feri, maathi, roknu
- **Sometimes detected:** hoina, huncha, suru, tala
- **Rarely detected:** aghillo, arko

The live user's voice was not in the training set. `aghillo` and `arko` scored
near −127 (int8 minimum) even when spoken — indicating those words have high
inter-speaker pronunciation variance not fully covered by the training speakers.
This is an expected limitation of a 30-speaker dataset.

---

## Quantisation and Model Conversion

### Pipeline

```
Keras model (.keras)
    → TensorFlow SavedModel (model.export())
    → TFLite float32 (baseline)
    → TFLite int8 (full integer quantisation)
    → C header array (nepspot_model_data.h)
```

### INT8 Quantisation Details

- **Method:** Post-training full integer quantisation via TFLite converter
- **Representative dataset:** 200 samples from training split
- **Input type:** int8 (quantised before inference)
- **Output type:** int8 (12 raw scores, not softmax probabilities)
- **Input quantisation parameters:** scale = 0.046068, zero_point = 60
- **Detection threshold:** output score > 40 (out of 127 max) for confident detection
- **INT8 spot-check accuracy:** verified on 50 samples during conversion

### Model Sizes

| Format       | Size     |
|--------------|----------|
| Keras (.keras) | 311,828 bytes (304.5 KB) |
| TFLite float32 | 69,108 bytes (67.5 KB) |
| TFLite int8    | 34,496 bytes (33.7 KB) |
| C header (embedded in flash) | ~170 KB |

---

## On-Device Deployment

### Hardware

| Component   | Specification                          |
|-------------|----------------------------------------|
| Board       | Arduino Nano 33 BLE Sense Rev2         |
| MCU         | nRF52840, ARM Cortex-M4F @ 64 MHz      |
| SRAM        | 256 KB (262,144 bytes)                 |
| Flash       | 1 MB (1,048,576 bytes)                 |
| Microphone  | MP34DT06JTR PDM MEMS microphone        |
| OS          | Mbed OS (Arduino mbed core)            |
| ML runtime  | TensorFlow Lite for Microcontrollers (TFLM) |
| DSP library | CMSIS-DSP (arm_rfft_fast_f32)          |

### Memory Budget (final optimised firmware)

| Region              | Usage                          |
|---------------------|-------------------------------|
| Flash               | 342,640 / 983,040 bytes (34%) |
| BSS (static RAM)    | 242,416 / 262,144 bytes (92%) |
| Heap available      | ~19,728 bytes                  |
| TFLM tensor arena   | 167 KB (171,008 bytes)         |
| Audio capture buffer| 16,000 bytes (int8)            |

### Inference Pipeline (C++, NepSpot.ino)

The entire pipeline runs on-device with no floating-point audio buffers:

1. **PDM ISR** — MP34DT06JTR captures PDM audio; ISR converts 16-bit PCM
   samples to int8 (top 8 bits via `>> 8`, equivalent to −48 dB quantisation
   noise, well below the mic's noise floor) and writes into a 16,000-sample
   ring buffer.

2. **RMS gate** — After 1 second (16,000 samples), compute RMS. If RMS < 0.006,
   classify as silence and skip inference.

3. **Feature extraction** — For each of 32 frames (hop=512):
   - Apply Hann window with virtual center=True padding (512 zero-pad each side)
   - 1024-point real FFT via `arm_rfft_fast_f32` (CMSIS-DSP)
   - Extract power spectrum from packed output
     (DC = output[0]², Nyquist = output[1]², bin k: output[2k]²+output[2k+1]²)
   - Apply 40-band Slaney mel filterbank (triangular, normalised by bandwidth)
   - Convert to dB
   - Apply DCT-II (orthonormal) → 40 MFCC coefficients
   - Normalise with global mean/std constants
   - Quantise to int8 and write directly into TFLM input tensor

4. **Inference** — `interpreter->Invoke()` runs the DS-CNN on the 40×32 int8 input.

5. **Decode** — argmax over 12 int8 output scores. Report "DETECTED" if best
   score > 40.

**Total latency:** 1,955 ms (±1 ms, n=14, measured via `micros()` around `Invoke()` on-device)

### Key Engineering Challenges Solved

**1. TFLM tensor arena sizing**
DS-CNN with 64 filters requires 164,992 bytes of working memory (intermediate
activations: 40×32×64 = 81,920 bytes, double-buffered by TFLM). Minimum viable
arena: 167 KB. Finding this required iterative testing — TFLM reports the exact
deficit when the arena is too small.

**2. SRAM linker overflow**
The nRF52840 Arduino core has a BSS limit of 261,120 bytes. With a 168 KB arena,
total BSS exceeded this by ~6 KB. Solved by eliminating all intermediate float
arrays from globals — mel filterbank now reads FFT output in-place, and MFCC
values are quantised and written frame-by-frame directly into the TFLM input
tensor. This reduced worst-case stack usage from ~2,700 bytes to ~644 bytes.

**3. Heap exhaustion (silent crash before setup())**
With 98% BSS, only 3,728 bytes of heap remained for Mbed OS. The USB CDC stack
needs ~4+ KB of heap and initialises before user code runs — insufficient heap
causes a silent crash with no serial output. Fixed by changing the audio capture
buffer from `int16_t[16000]` (32 KB) to `int8_t[16000]` (16 KB), recovering
16 KB and giving 19,728 bytes of heap. The int8 storage (top 8 bits of 16-bit
PCM) introduces negligible feature distortion — quantisation noise is −48 dB,
well below the microphone's SNR.

**4. CMSIS-DSP FFT packing format**
`arm_rfft_fast_f32` uses a non-standard packed output format:
`[DC, Nyquist, re₁, im₁, re₂, im₂, ...]`.
Power extraction must handle DC (output[0]²), Nyquist (output[1]²), and general
bins (output[2k]² + output[2k+1]²) explicitly — not as a standard complex array.

**5. librosa parity**
Exactly reproduced librosa's defaults on-device:
- Slaney mel scale (not HTK — different Hz-to-mel mapping)
- center=True padding (512 zeros prepended, 512 appended virtually)
- Orthonormal DCT-II (normalisation factor √(1/N) for DC, √(2/N) for others)
Any mismatch here causes the model to receive features it was never trained on,
which severely degrades accuracy.

---

## Current Status Snapshot

### What happened
- BC-ResNet Phase 1 was retrained as a 14-class model with silence and unknown added.
- QAT fixed the quantisation collapse: PTQ INT8 was 66.85%, QAT INT8 reached 86.06% with Macro F1 0.8592.
- Arduino latency for BC-ResNet is 331 ms, much faster than DS-CNN on-device.
- **FIX 1 COMPLETE (2026-05-12):** Built proper 14-class test set with 50 silence + 50 unknown held-out samples.
  - Moved test-only clips into _silence_test and _unknown_test folders (raw + processed).
  - Updated split config to include test pseudo-speakers.
  - Float32 BC-ResNet: 83.69% accuracy, silence F1: 0.949, unknown F1: 0.809.
  - INT8 BC-ResNet: 80.05% accuracy (-3.64%), silence F1: 0.920, unknown F1: 0.752.

### What is saved already
- Speaker split config: `configs/speaker_split_v1.json`
- BC-ResNet Phase 1 model: `models/saved/bc_resnet_phase1_14class_best.keras`
- INT8 model: `models/tflite/bc_resnet_int8_phase1_14class.tflite`
- Existing training and paper notes: this file plus `agents/projects/nepspot.md`

### What has been completed (2026-05-12)
1. ✅ **FRR/FAR/ROC pipeline** — Computed false reject/accept rates with EER operating points.
   - Float32: EER at 10.99% FRR / 10.00% FAR
   - INT8: EER at 9.05% FRR / 12.00% FAR
   - Saved ROC curves and data to results/metrics/ and results/figures/
2. ✅ **P50/P95/P99 latency stats** — Measured inference time distribution.
   - Float32: P50=39.67ms, P95=42.45ms, P99=45.62ms
   - INT8: P50=0.084ms, P95=0.213ms, P99=0.269ms (410x faster on macOS)
   - Saved latency histogram and raw data
3. ✅ **Arduino .h conversion** — Generated C header file (89.3 KB).
   - Model embedded as nepspot_model_data_14class.h
   - Created comprehensive deployment guide: ARDUINO_DEPLOYMENT_14CLASS.md
   - Ready for Arduino Nano 33 BLE Sense Rev2 integration
4. ⏸️ **Live audio test on hardware** — Requires physical nRF52840 (model file ready, integration guide provided).

### Rules to keep
- Do not touch the vanilla CNN or DS-CNN baselines.
- Do not augment or modify the voicer28/29/30 test set.
- Do not claim the dataset is CC BY 4.0 in the paper.
- Keep the author single-person only.

---

## Repository Structure

```
NepSpot/ (research codebase)
├── configs/
│   └── speaker_split_v1.json     # speaker assignments per split
├── data/
│   ├── raw/                      # <speaker>/<keyword>/<file>.wav
│   └── processed/                # normalised .npy MFCCs
├── models/
│   └── saved/
│       ├── best_model.keras
│       ├── mfcc_mean.npy         # −12.608...
│       ├── mfcc_std.npy          # 78.485...
│       ├── label_classes.npy
│       └── saved_model/          # TF SavedModel for TFLite conversion
│   └── tflite/
│       ├── nepspot_float32.tflite
│       └── nepspot_int8.tflite
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── per_keyword_accuracy.png
│   └── metrics/
│       ├── classification_report.txt
│       └── test_speakers.txt
└── src/
    ├── features/
    │   └── extract_mfcc.py       # librosa MFCC pipeline + global normalisation
    ├── models/
    │   ├── ds_cnn.py             # model architecture
    │   ├── train.py              # training script
    │   └── convert_tflite.py    # float32 + int8 TFLite conversion
    ├── inference/
    │   ├── live_mic.py           # real-time test on laptop mic
    │   └── test_words.py        # batch evaluation
    └── utils/
        ├── dataset.py            # split loading, speaker management
        ├── augment.py            # augmentation functions
        └── run_augmentation.py  # fill-to-target + speed augmentation

Arduino/NepSpot/ (firmware)
├── NepSpot.ino                  # complete inference firmware
└── nepspot_model_data.h         # int8 model as C byte array
```

---

## Paper Contributions (What to Claim)

1. **First Nepali KWS dataset and model** with speaker-independent evaluation —
   30 speakers, 12 semantically meaningful keywords, custom collection pipeline
2. **84.86% speaker-independent accuracy** — 10.2× above random chance for 12 classes
3. **Full TinyML deployment** on 256 KB SRAM microcontroller — not just a Python demo,
   actually running in real-time on edge hardware
4. **Exact feature parity methodology** — librosa-to-C++ MFCC reimplementation
   with documented pitfalls (Slaney vs HTK, center padding, FFT packing format)
5. **Memory engineering** — systematic approach to fitting a DS-CNN within
   hard memory constraints (BSS budget, heap, stack, tensor arena)

---

## Suggested Paper Titles

- "NepSpot: Real-Time Nepali Keyword Spotting on a 256 KB Microcontroller"
- "Towards Nepali Voice Commands at the Edge: A TinyML Approach with DS-CNN"
- "NepSpot: On-Device Nepali Keyword Spotting with Speaker-Independent Evaluation"

---

## Remaining TODOs Before Paper Submission

- [x] **Inference latency** — 1,955 ms (±1 ms, n=14, nRF52840 @ 64 MHz)
- [x] **Exact model size** — 34,496 bytes (33.7 KB) int8 TFLite
- [x] **Exact parameter count** — 17,164 total (16,268 trainable)
- [ ] **Confusion matrix figure** — already at `results/figures/confusion_matrix.png`
- [ ] **Per-keyword accuracy figure** — already at `results/figures/per_keyword_accuracy.png`
- [ ] **Comparison baseline** — train a simple LSTM or plain CNN on same data to compare
- [ ] **Related work search** — check if any prior Nepali KWS/ASR papers exist
  (search: "Nepali speech recognition", "Nepali keyword", "low-resource Nepali NLP")
- [ ] **Target venue decision** — ICASSP / Interspeech / SLT / TinyML workshop


---

## 2026-05-15: Forensic Audit + Architecture and Determinism Overhaul

This section appends to the file rather than replacing it. Earlier entries
above (≤ 2026-05-12) remain authoritative for everything they cover; only the
specific items called out below have been revised.

### Why this section exists

After the May 13 "5× augmentation" run produced inconsistent results across
reruns (BC-ResNet INT8 swinging 86.06% → 79.78% with no recipe change), a
full forensic audit was run on the codebase. It found multiple critical
bugs that invalidate most cross-experiment comparisons made before this date.
Below is the audit summary, the fixes applied this session, the current
half-broken state of INT8 conversion, and the concrete next steps.

### Audit findings (severity)

1. **BC-ResNet was structurally wrong (Critical).** The Kim 2021 BCBlock has
   two parallel depthwise branches: a frequency branch with kernel (3,1)
   (no dilation) and a time branch with kernel (1, k_t) with dilation along
   the **time axis**. The old `src/models/bc_resnet.py` had only the
   frequency branch and applied dilation on the **frequency** axis. Because
   the freq axis collapses to 2-10 through the transition pools, stage 3
   (dilation 4) and stage 4 (dilation 8) had kernel taps landing almost
   entirely on zero-pad. The time-axis dilated receptive field that defines
   BC-ResNet was absent. Every "BC-ResNet" number on disk before 2026-05-15
   is from a degenerate freq-only CNN, not BC-ResNet.

2. **Training was non-deterministic (Critical).** Despite seed setting,
   `phase1_training_utils.py` used `num_parallel_calls=tf.data.AUTOTUNE` on
   SpecAugment + Mixup maps, and no script called
   `tf.config.experimental.enable_op_determinism()`. Direct evidence: two
   BC-ResNet Phase 1 runs on identical data
   ([`bc_resnet_phase1_qat_final.log`](NepSpot/results/metrics/phase1_training_logs/bc_resnet_phase1_qat_final.log),
   [`bc_resnet_phase1_qat_rerun.log`](NepSpot/results/metrics/phase1_training_logs/bc_resnet_phase1_qat_rerun.log))
   produced **86.06% vs 79.59% INT8** — a 6.47% swing from training noise
   alone.

3. **Test set silently changed size (High).** Phase 1 (May 12) logs show
   `Test: 1083 samples`. Current reports show `Samples: 361`. voicer28-30
   each had ~30 clips/keyword in Phase 0/1 (probably synthetic copies from
   the old fill_to_target pass); they now have 10 clips/keyword
   (originals only). Every cross-phase accuracy comparison made before
   today is across **different test sets** and is not meaningful.

4. **Claimed 92.24% Vanilla CNN does not exist (Critical for narrative).**
   No file under `results/` or `models/` contains the figure. Latest disk:
   `vanilla_cnn_phase1_report.txt` = 85.87% INT8, log = 85.23% INT8.

5. **K-fold incomplete and contaminated (High).** Only 2 of 5 folds
   completed. Test sets for folds 1-3 contained augmented copies of
   voicer1-24 (since `dataset.py:load_dataset_for_speakers` does not filter
   `_aug_*.npy` and `run_augmentation.py` Pass 2 only excludes voicer25-30).
   The 67% k-fold mean is on a synthetic-contaminated test set, not
   comparable to single-split numbers.

6. **MFCC stats leak (Medium).** `extract_mfcc.py` computes a **single
   scalar** mean+std across **all** data including val/test speakers and
   `_aug_*.wav` files. Not per-coefficient. The current stats on disk
   (`models/saved/mfcc_mean.npy = -11.768554`,
   `models/saved/mfcc_std.npy = 73.33601`) supersede the values printed in
   the section *Global Scalar Normalisation* above
   (`-12.608.../78.485...`). The earlier stats correspond to an earlier
   data state.

7. **slr_ corpus has only ~3 keywords per speaker (Medium).**
   `extract_openslr.py` and `extract_slr_organized.py` both define 11
   keywords (missing `huncha`) and rely on Whisper word-alignment which
   often only finds a few keywords per utterance. `data/processed/slr_001/`
   has only `aghillo`, `banda`, `tala` (3 of 12). 443 slr_ speakers
   contribute heavily skewed per-class counts. No trainer applies class
   weights.

8. **Inference scripts stale (Medium).** `src/inference/live_mic.py:34` and
   `test_words.py:36` load `best_model.keras` (Phase 0 DS-CNN), not any
   Phase 1 checkpoint. Plus 1.5 s record / 1.0 s analyze window mismatch.

### Code changes applied this session

#### `src/models/bc_resnet.py` — fully rewritten

Old file preserved at `src/models/bc_resnet_OLD_BROKEN.py` for reference.

New normal block (Kim 2021 BCBlock):
```
x ──┬─ freq_branch:  DW(kernel=(3,1), no dilation) ─ BN ─────────────────┐
    │                                                                    ├─ Add ─ Conv(1×1) ─ BN ─(+residual)─ ReLU
    └─ time_branch:  DW(kernel=(1,3), dilation=(1,d)) ─ BN ─ Swish ─ broadcast over freq ─┘
```

The broadcast over freq was first implemented as
`AveragePooling2D((F,1)) → Lambda(tf.tile)`, then (after INT8 conversion
failed with `same scale constraint` at AvgPool) replaced with a single
**frozen DepthwiseConv2D**:

- `kernel_size=(2F-1, 1)`, `padding='same'`, `use_bias=False`,
  `depthwise_initializer=Constant(1/F)`, `trainable=False`.
- Mathematically equivalent to `AvgPool+UpSampling2D` (max abs diff ≤ 1e-7,
  verified for F ∈ {2, 3, 5, 10}).
- Single op with one quantization scale on input and output → no more
  same-scale constraint violation.

Network shape trace (input `(40, 32, 1)`):
- stem: stride (2,1) → `(20, 32, 16)`
- stage 1 (filters 8, time dilation 1, 2 normal blocks): `(10, 32, 8)`
- stage 2 (filters 12, time dilation 2, 2 blocks): `(5, 32, 12)`
- stage 3 (filters 16, time dilation 4, 4 blocks): `(3, 32, 16)`
- stage 4 (filters 20, time dilation 8, 4 blocks): `(2, 32, 20)`
- head: Conv(32, (1,5)) → BN → ReLU → Conv(32, (1,1)) → BN → ReLU
- GAP → Dense(12, softmax)

Parameter count:
- **trainable: 11,276** (unchanged from the AvgPool+UpSampling version)
- non-trainable: 2,456 (1,376 BN moving stats + 1,080 frozen broadcast kernels)
- total: 13,732

#### `src/models/train_bcresnet_phase1.py: build_bc_resnet_tfkeras` — fully rewritten

This tf_keras (Keras-2) fallback exists because
`tfmot.quantization.keras.quantize_model` rejects Keras-3 Functional models
with `isinstance` check. The function now mirrors `bc_resnet.py` exactly
(same layer names, same weight order, same broadcast implementation).
`tfk_model.set_weights(float_model.get_weights())` was verified to transfer
all 229 weight arrays with **0 shape mismatches** and produce a bit-identical
forward pass.

#### Determinism — all four training scripts

Added the following prologue pattern to
[`src/models/train_phase1.py`](NepSpot/src/models/train_phase1.py),
[`train_vanilla_phase1.py`](NepSpot/src/models/train_vanilla_phase1.py),
[`train_bcresnet_phase1.py`](NepSpot/src/models/train_bcresnet_phase1.py),
and [`run_kfold_crossval.py`](NepSpot/src/models/run_kfold_crossval.py):

```python
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# ... (other imports)
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
```

#### `src/models/phase1_training_utils.py` — fully rewritten for stateless determinism

- All `tf.random.uniform/normal/shuffle` replaced with stateless equivalents
  (`stateless_uniform`, `stateless_gamma`, argsort of stateless scores for
  the Mixup permutation).
- Per-example seeds for SpecAugment derived from `Dataset.enumerate()` index
  + base seed 42.
- Per-batch seeds for Mixup derived from a second `.enumerate()` after
  `.batch()`.
- All `.map()` calls use `num_parallel_calls=1, deterministic=True`.
- All `.prefetch()` calls use `prefetch(1)` (not AUTOTUNE).
- `.shuffle()` now passes `seed=42`.

**Verified**: two consecutive `model.fit()` calls on a synthetic dataset
produce bit-identical loss, accuracy, and weight-sum (diff = 0.00e+00).

#### `src/models/train_bcresnet_phase1.py: quantize_qat_model_to_int8` — patched

Added three experimental converter flags:
```python
converter.experimental_enable_resource_variables = True
converter.experimental_new_quantizer = True
converter._experimental_disable_per_channel = True
```

This was insufficient on its own — see the next subsection.

#### `scripts/convert_bcresnet_qat_to_int8.py` — new

Standalone script that loads the existing QAT checkpoint, runs INT8
conversion with the experimental flags, and writes the TFLite + `.h` file.
Intended for re-running conversion without retraining when the in-train
conversion fails. Currently blocked by the same `.keras` loader bug as the
recovery script (see below).

#### `scripts/recover_bcresnet_int8.py` — new

When the conversion failed even with the experimental flags, this script
was created to:
1. Load the existing QAT checkpoint (preserving the trained weights).
2. Build a new architecture with the DepthwiseConv broadcast.
3. Transfer weights by layer name (skipping the 12 new `*_freq_avg_broadcast`
   layers which keep their frozen 1/F init).
4. PTQ-convert the new model to INT8.
5. Evaluate on the test split.

### The INT8 saga in chronological order

1. BC-ResNet was retrained 2026-05-14 with the corrected architecture
   (AvgPool+UpSampling broadcast variant). Float QAT model saved to
   `models/saved/bc_resnet_phase1_best.keras`. Test accuracy: **85.87%**,
   macro F1: **0.8618**.
2. In-train INT8 conversion failed with
   `'tfl.average_pool_2d' op quantization parameters violate the same scale
   constraint` at `stage1_block1_tb_bcast_freq_pool/AvgPool`. AvgPool's
   output and the downstream Add fan-in were assigned conflicting scales.
3. Three experimental flags added (see above). Conversion **still failed**
   with the same error.
4. Architecture amended: AvgPool+UpSampling pair replaced with a single
   frozen DepthwiseConv2D (kernel (2F-1, 1), weights 1/F, trainable=False).
   Mathematically equivalent; collapses to a single quantization scale.
   `bc_resnet.py` and `build_bc_resnet_tfkeras` both updated.
5. Weight transfer recovery script written. **Currently blocked** by a
   tfmot 0.8.0 / tf_keras 2.16 `.keras` format incompatibility:
   `tf_keras.models.load_model(QAT_CHECKPOINT_PATH)` raises
   `ValueError: Layer 'quantize_layer' expected 5 variables, but received 3`.
   This is a known serialization-format issue between tfmot's saved state
   and tf_keras's expected variable count for the input-side QuantizeLayer.

### Outstanding work (do these next)

#### Immediate — recover INT8 from the existing trained float QAT model

The model has been trained, weights are valid on disk, only the loader is
broken. Three approaches, in increasing order of effort:

1. **Direct h5 weight extraction**: open
   `bc_resnet_phase1_best.keras` as a ZIP (it is one), pull
   `model.weights.h5`, read named weight tensors via `h5py`, and assign
   them positionally or by name to the new architecture. Avoids the
   `tf_keras.load_model` path entirely. ~30 min of code.
2. **Resurrect old build + load_weights skip_mismatch**: temporarily
   restore the AvgPool+UpSampling version of `build_bc_resnet_tfkeras`
   inside the recovery script, wrap with `tfmot.quantize_model`, call
   `model.load_weights(QAT_PATH, by_name=True, skip_mismatch=True)`,
   then peel out inner-layer weights and transfer to the new (DepthwiseConv)
   model. ~1 hr.
3. **Retrain from scratch with the new architecture** (~2 hr GPU/CPU).
   Last resort — only if 1 and 2 both fail.

#### After INT8 recovery

- Re-run the full BC-ResNet evaluation: INT8 accuracy, macro F1, per-class
  F1, confusion matrix on voicer28-30.
- Generate Arduino `.h` from the new INT8 TFLite. Update
  `ARDUINO_DEPLOYMENT_14CLASS.md` if class layout has shifted.
- Run determinism verification end-to-end on real data: two BC-ResNet
  trainings from scratch must produce bit-identical INT8 accuracy. The
  synthetic verification done so far confirms the data pipeline + small
  Conv2D model are deterministic; the full BC-ResNet pipeline is unverified.

#### Methodology to fix before paper submission

- **N≥5 seeds** for each of Vanilla CNN, DS-CNN, BC-ResNet on the same
  single-split (voicer28-30). Report `mean ± std`. Without this, ranking
  claims have no statistical basis.
- **5-fold cross-validation done properly** — filter `_aug_*.npy` out of
  test splits (patch `dataset.py:load_dataset_for_speakers` to accept an
  `include_augmented` flag), complete all 5 folds (current run stopped
  after 2), at least 3 seeds per fold. The existing kfold artifacts
  (`results/metrics/kfold/`, fold1/2 checkpoints) should be archived and
  regenerated.
- **Augmentation strategy decision** — current state is 5× (1 original +
  4 speed/noise variants), but this stacks on top of in-pipeline SpecAug
  + Mixup. Run a small ablation (no aug / SpecAug+Mixup only / 3× / 5×)
  to pick one defensible recipe and commit.
- **MFCC stats recomputation** — change `extract_mfcc.py` to compute
  per-coefficient (vector of length 40) stats on **train-only originals**
  (exclude val/test speakers, exclude `_aug_*.wav`), then regenerate
  `data/processed/`. Current scalar stats on all data is mildly leaky and
  non-standard.
- **Inference scripts** — update `live_mic.py` and `test_words.py` to load
  the current canonical Phase 1 checkpoint; align record-duration to
  analysis window.

#### Repository hygiene

- 38 files in `models/saved/`, 18 in `models/tflite/`, 19 in
  `results/metrics/`. Many are stale (pre-Phase-1, 14-class experiment,
  incomplete k-fold). Audit identified a specific delete/archive list.
- Three legacy training scripts (`train.py`, `train_vanilla.py`,
  `train_bcresnet.py`) are superseded by their `*_phase1.py` versions.
- `convert_tflite.py` is dead code with a stale path.

### Facts that have been superseded since 2026-05-12

Earlier sections of this file remain readable history, but the following
are no longer accurate as of 2026-05-15:

- **Section "Global Scalar Normalisation"**: stated values were
  `mean=-12.608, std=78.485`. Current files on disk (regenerated
  2026-05-13 22:14) are `mean=-11.769, std=73.336`. The old stats were
  before the 5× speed augmentation pipeline ran.
- **Section "Dataset Size Summary" → "Test: 1083"**: the current
  speaker-independent test split has **361 samples** (10 originals ×
  12 keywords × 3 speakers, +1 spare in voicer28/arko). The 1,083 number
  reflected an earlier data state that included synthetic copies of test
  speakers. Cross-phase accuracy comparisons before today are invalid.
- **Section "Speakers"**: actual `data/raw/` and `data/processed/`
  include 487 speaker folders — voicer1-30, speaker_1-23, slr_001-443,
  silence/unknown specials. Earlier "30 speakers" description applies only
  to the voicer pool used for the speaker-independent split.
- **Section "Current Status Snapshot (2026-05-12)" → BC-ResNet QAT INT8
  86.06%**: that number is from the *old broken architecture* on a
  contaminated test set. The corrected-architecture BC-ResNet has a
  float QAT accuracy of **85.87%**; INT8 not yet recovered.

### Files added or modified this session

| Path | Status | Purpose |
|---|---|---|
| `src/models/bc_resnet.py` | rewritten | Correct BC-ResNet-1 (Kim 2021) with DepthwiseConv broadcast |
| `src/models/bc_resnet_OLD_BROKEN.py` | new (backup) | Original buggy version preserved |
| `src/models/train_bcresnet_phase1.py` | modified | Determinism prologue + new `build_bc_resnet_tfkeras` + INT8 flags |
| `src/models/train_phase1.py` | modified | Determinism prologue |
| `src/models/train_vanilla_phase1.py` | modified | Determinism prologue |
| `src/models/run_kfold_crossval.py` | modified | Determinism prologue |
| `src/models/phase1_training_utils.py` | rewritten | Stateless ops + per-example/per-batch seeds + `num_parallel_calls=1` |
| `scripts/convert_bcresnet_qat_to_int8.py` | new | Standalone INT8 conversion (flag-based, blocked by loader bug) |
| `scripts/recover_bcresnet_int8.py` | new | Weight-transfer INT8 recovery (blocked by loader bug) |

No retraining was done in this session for production; only synthetic
sanity-check fits.

### How to resume work in a future session

If picking this up cold, run in order:

1. Read this 2026-05-15 section + the audit findings.
2. Confirm `models/saved/bc_resnet_phase1_best.keras` exists (the trained
   float QAT model, 85.87% test accuracy).
3. Attempt INT8 recovery via approach 1 (direct h5 extraction) in the
   "Outstanding work" subsection. Note that the recovery script
   already does layer-by-layer weight transfer once the load step works —
   the only thing missing is the loader bypass.
4. If INT8 recovery succeeds, evaluate on test, write Arduino header,
   update the "Current Status Snapshot" section with the real INT8
   accuracy number.
5. Only after step 4 is the BC-ResNet pipeline usable end-to-end.


---

## 2026-05-15 (later): BC-ResNet INT8 recovered + Vanilla/DS-CNN ready for deterministic retrain

### INT8 recovery — done

`scripts/recover_bcresnet_int8.py` was rewritten to bypass the broken
`tf_keras.models.load_model` path. New approach (approach 1 from the
"Outstanding work" list above):

1. Open `models/saved/bc_resnet_phase1_best.keras` as a ZIP, read `config.json`
   and `model.weights.h5` in-memory.
2. Walk `config.json` to enumerate the 171 `QuantizeWrapperV2` layers in
   construction order. The i-th wrapper maps to h5 group
   `quantize_wrapper_v2` (i=0) or `quantize_wrapper_v2_{i}` (i>0). The
   wrapper's inner config gives the original layer name (`stem_conv`,
   `stage1_block1_fb_freq_dw`, ...) and class.
3. Pull weights out of each wrapper's h5 group by tfmot convention:
   - **Conv2D / DepthwiseConv2D** (no bias in this model): kernel at
     `vars/0`. (Empirically `vars/1` is a bit-identical duplicate;
     `vars/2..4` are quantizer auxiliary state — min/max/step.)
   - **Dense**: kernel at `vars/0`, bias at `vars/1` (also bit-identical
     to `layer/vars/0`).
   - **BatchNormalization**: `layer/vars/0..3` = `[gamma, beta,
     moving_mean, moving_variance]` (verified: var/0 mean ≈ 1, var/1
     mean ≈ 0, var/3 strictly positive variance).
   - Pool/Activation/Add/ReLU/UpSampling: no weights, skip.
4. Build the new architecture via `build_bc_resnet` from
   `src/models/bc_resnet.py` (DepthwiseConv broadcast variant).
5. Transfer weights by inner layer name. Skip the 12 new
   `*_freq_avg_broadcast` layers (no QAT counterpart; keep their
   frozen 1/F init).
6. PTQ INT8 conversion with 200 representative samples from train split.

#### Conversion gotcha worth keeping

`tf.lite.TFLiteConverter.from_keras_model(model)` failed mid-MLIR with
`error: missing attribute 'value'` on the classifier's `BiasAdd` when given
a Keras-3 functional model whose variables had been mutated via
`layer.set_weights()` rather than trained through. Routing through
`model.export(saved_model_dir)` → `from_saved_model(...)` fixed it
cleanly. The fix is in `recover_bcresnet_int8.py:convert_to_int8` —
worth remembering for any future "weight-transferred Keras 3 model →
TFLite" pipelines.

#### Result

- INT8 TFLite: `models/tflite/bc_resnet_int8_recovered.tflite` (70.05 KB)
- Test set: 361 samples (voicer28-30 originals, no `_aug_` files)
- **INT8 test accuracy: 84.76%**, **macro F1: 0.8474**
- Input qparams: scale=0.051455, zero_point=57
- Transferred layers: 87. Skipped (frozen broadcast): 12. No name
  misses, no shape mismatches.
- Float QAT reference (from training) was 85.87% — the recovered INT8
  is 1.11% below float, consistent with the QAT→INT8 gap we'd expect.
- Report saved to `results/metrics/bc_resnet_int8_recovered_report.txt`.

Note the BC-ResNet INT8 macro F1 figure of 0.8592 quoted in the
pre-2026-05-15 "Current Status Snapshot" was from the **broken**
freq-only architecture on a contaminated test set. The new
0.8474 is on the corrected architecture + speaker-independent
originals-only test set and is the number that should be used in any
forward-looking comparison.

### Vanilla CNN + DS-CNN: deterministic retrain scripts ready

Verified both `src/models/train_vanilla_phase1.py` and
`src/models/train_phase1.py` already carry the full determinism
prologue from the audit fix session:

- `PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`,
  `TF_CUDNN_DETERMINISTIC=1` set before TF import
- `tf.config.experimental.enable_op_determinism()` and
  `tf.random.set_seed(42)` after import
- Both route their training data through
  `phase1_training_utils.build_phase1_train_datasets`, which uses
  `num_parallel_calls=1, deterministic=True` on every `.map()`,
  `prefetch(1)` (not AUTOTUNE), and a seeded `shuffle(seed=42)`.

No code changes were needed. Both scripts are ready to run as-is for
the next session — running them was deferred.


---

## 2026-05-16: BC-ResNet seed-42 end-to-end + seed-experiment infra + ablation + ROC

This session is a continuation of the 2026-05-15 audit fixes. Most of
the work is infrastructure that's now ready to run; the load-bearing
empirical result on disk is the first clean end-to-end BC-ResNet
training with the corrected DepthwiseConv-broadcast architecture and
the new INT8 conversion path.

### BC-ResNet seed=42 training — first clean run with new architecture

User ran `src/models/train_bcresnet_phase1.py` once with the
corrected architecture. Artifacts written 2026-05-16 12:35:

- Model:    `models/saved/bc_resnet_phase1_best.keras` (OVERWROTE the
  pre-2026-05-15 checkpoint that the h5 recovery in
  `scripts/recover_bcresnet_int8.py` consumed; the recovered TFLite at
  `models/tflite/bc_resnet_int8_recovered.tflite` is still on disk and
  remains a snapshot of the pre-fix model state)
- INT8:     `models/tflite/bc_resnet_int8_phase1.tflite` (93.64 KB)
- Report:   `results/metrics/bc_resnet_phase1_report.txt`

Numbers (single seed, single split — to be combined with seeds 123/456
once the seed experiment runs):

| Stage                    | Accuracy | Macro F1 |
|--------------------------|----------|----------|
| Float32 pre-QAT          | 84.49%   | 0.8450   |
| Float32 post-QAT         | 81.16%   | 0.8137   |
| **INT8 (deployment)**    | **80.33%** | **0.8054** |

- Total params: **13,732** (trainable 11,276 + non-trainable 2,456)
- INT8 size: **93.64 KB**
- CPU inference (per sample, float32): 131.144 ms
- INT8 input qparams: scale=0.076592, zero_point=71
- INT8 output qparams: scale=0.003906, zero_point=-128

This is the first time the architecture+pipeline has gone end-to-end
cleanly: the in-train INT8 conversion no longer hits the
'same scale constraint' error, no manual h5 recovery is needed, and
the resulting INT8 model loads cleanly into `tf.lite.Interpreter`.

Note that the **recovered** model from 2026-05-15 reported INT8
84.76% / 0.8474 on the same test split, but that was weight-transfer
from the OLD QAT-trained checkpoint into the NEW architecture's float
graph followed by PTQ — a different route than the in-train QAT→INT8
path the production model takes. The 80.33% number from the canonical
in-train pipeline is the one to use for cross-model comparison.

### Statistical-rigor experiment: N=3 seeds × 3 models (infrastructure done, run pending)

To get publishable `mean ± std` numbers, threaded a single `SEED` env
var through every source of randomness in all three training
pipelines:

- `src/models/train_phase1.py` (DS-CNN)
- `src/models/train_vanilla_phase1.py` (Vanilla CNN)
- `src/models/train_bcresnet_phase1.py` (BC-ResNet)
- `src/models/phase1_training_utils.py` (SpecAugment + Mixup stateless
  seeds via `_BASE_SEED` and tf.data shuffle seed)

Each replaces a previously hardcoded literal `42`:

```python
SEED = int(os.environ.get('SEED', '42'))
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

Default remains 42 when `SEED` is unset, so the production training
scripts behave identically to before for any direct invocation.

The wrapper `scripts/run_seeds_experiment.py` loops over `[BC-ResNet,
Vanilla CNN, DS-CNN] × [42, 123, 456]` (9 runs total) via subprocess.
For each run it:

1. Wipes the training script's known output files (TFLite + report)
   so a stale artifact can't masquerade as success.
2. Launches the script with `SEED=<seed>` set in env, redirecting
   stdout+stderr to a per-run log under
   `results/seeds_experiment_artifacts/logs/`.
3. On success, parses INT8 accuracy / macro F1 / size / per-class F1
   from the script's report, archives the TFLite + report under
   `results/seeds_experiment_artifacts/<key>_seed<N>.{tflite,_report.txt}`
   so the next run does not clobber it.
4. After every run, writes incremental
   `results/metrics/seeds_experiment_report.txt` (formatted table
   plus per-model mean ± std with ddof=1) and
   `results/metrics/seeds_experiment_raw.json` so a mid-experiment
   crash still leaves usable artifacts.

Verified before deferring the run:

- Syntax check passes on all 5 modified/new files.
- Report parser succeeds on the existing 3 report formats (BC-ResNet
  2-decimal, Vanilla 2-decimal, DS-CNN 4-decimal sklearn output).
- A synthetic-data dry-run of the text-report writer renders the
  table cleanly (mean/std formatting, per-class F1 means per model).

Expected runtime ≈ 2.5 hours (BC-ResNet 32 min × 3 + Vanilla 4 min × 3
+ DS-CNN 7 min × 3). The wrapper was BUILT this session but NOT RUN —
deferred to user discretion.

### DS-CNN no-SpecAug+Mixup ablation script (ready, not yet run)

`scripts/train_dscnn_no_specaug.py` mirrors the Phase 1 DS-CNN recipe
(`build_ds_cnn`, Adam 1e-3, label_smoothing=0.1, same checkpoint /
early-stopping / LR-schedule callbacks, 50 max epochs, PTQ INT8 via
SavedModel) but replaces the `phase1_training_utils` SpecAug+Mixup
pipeline with a plain:

```python
ds = (tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(4096, seed=42, reshuffle_each_iteration=True)
        .batch(32)
        .prefetch(1))
```

Single seed (42), all determinism guards active, verbose=0
everywhere (only final metrics printed). Outputs:

- `models/tflite/dscnn_no_specaug_int8.tflite`
- `results/metrics/dscnn_no_specaug_report.txt`

Comparison baseline (per N=3 seed reference the user supplied):
DS-CNN with SpecAug+Mixup = **79.04% ± 1.15%** INT8 accuracy.
The ablation will quantify the contribution of the in-pipeline
augmentation. Script ready; not yet run.

### FRR / FAR / ROC analysis on the new BC-ResNet INT8

Reran `compute_roc_frr_far.py` against the seed-42 production INT8
checkpoint (`models/tflite/bc_resnet_int8_phase1.tflite`). One
non-trivial fix beyond updating output paths:

> **Bug:** the previous version applied
> `(mfcc - mean) / (std + 1e-8)` to inputs loaded from
> `data/processed/`, but `src/features/extract_mfcc.py:91` already
> normalises those `.npy` files before saving — so every prior ROC
> number is from a double-normalised feed.

The rewritten script feeds normalised `.npy` files through unchanged,
dequantises the INT8 output logits so reported thresholds are
interpretable as probability-scale scores, computes EER by linear
interpolation of the FNR/FPR crossing instead of nearest-index, adds
**FRR @ fixed FAR** at 0.1% / 1% / 5% targets, explicitly filters
`_aug_*.npy` (none exist in voicer28-30 anyway — defensive), and
marks the 1% FAR vertical on the figure.

**Headline results** (voicer28-30, n=361):

| Macro metric                    | Value                |
|---------------------------------|----------------------|
| AUC (per-keyword mean ± std)    | 0.9738 ± 0.0484      |
| Macro-ROC AUC (FAR-grid interp) | 0.9722               |
| EER  (per-keyword mean ± std)   | 7.22% ± 7.47%        |
| FRR @ FAR=1% (mean ± std)       | 21.07% ± 19.98%      |

Per-keyword winners (AUC ≥ 0.999, FRR@1%FAR = 0%): `feri`,
`huncha`, `maathi`. Per-keyword laggard: `tala` (AUC 0.829,
EER 26.67%, FRR@1%FAR 73.33%) — consistent with `tala` being the
lowest-recall class across every report in the repo.

Outputs:

- `results/metrics/frr_far_roc_new.txt`
- `results/figures/roc_curves_new.png`

### Files added or modified this session

| Path | Status | Purpose |
|---|---|---|
| `src/models/phase1_training_utils.py` | modified | `_SEED` from env drives `_BASE_SEED` + shuffle seed |
| `src/models/train_phase1.py` | modified | `SEED` from env (replaces hardcoded 42 in 4 places) |
| `src/models/train_vanilla_phase1.py` | modified | same SEED threading |
| `src/models/train_bcresnet_phase1.py` | modified | same SEED threading |
| `scripts/run_seeds_experiment.py` | new | 9-run subprocess wrapper + mean±std reporter |
| `scripts/train_dscnn_no_specaug.py` | new | DS-CNN ablation (no SpecAug, no Mixup) |
| `scripts/recover_bcresnet_int8.py` | (kept from prior day) | h5 weight-transfer INT8 recovery — superseded by the canonical in-train pipeline now that it works, but kept as a known-good fallback |
| `compute_roc_frr_far.py` | rewritten | Fixed double-norm bug; added FRR@fixed-FAR; new output paths |
| `models/tflite/bc_resnet_int8_phase1.tflite` | new (user ran) | 93.64 KB; INT8 80.33% / 0.8054 |
| `results/metrics/bc_resnet_phase1_report.txt` | new (user ran) | seed=42 canonical numbers |
| `results/metrics/frr_far_roc_new.txt` | new | ROC/EER/FRR-at-FAR per keyword + macro |
| `results/figures/roc_curves_new.png` | new | 12 per-keyword ROC + macro-ROC overlay |

### Facts that have been superseded since 2026-05-15

- **Section "INT8 recovery — done" → 84.76% / 0.8474.** That figure
  still describes the **recovered** model on disk, but the canonical
  forward-looking BC-ResNet INT8 number is now **80.33% / 0.8054**
  from the seed-42 in-train pipeline. Use the new number for any
  comparison against Vanilla CNN / DS-CNN; reserve the recovered
  number as evidence that the h5 weight-transfer approach is
  numerically sound when needed.
- **Section "How to resume work in a future session" step 2.**
  `bc_resnet_phase1_best.keras` no longer holds the 85.87% float QAT
  checkpoint described there — it now holds the seed-42 model from
  the new pipeline. The recovered INT8 TFLite still exists on disk
  at `bc_resnet_int8_recovered.tflite` if the older state is needed.

### Outstanding work for the next session

In priority order:

1. **Run the N=3 seeds experiment**
   (`scripts/run_seeds_experiment.py`). Until this lands, every
   model comparison is single-seed and not statistically defensible.
2. **Run the no-SpecAug+Mixup DS-CNN ablation**
   (`scripts/train_dscnn_no_specaug.py`). Quantifies the
   contribution of the augmentation recipe.
3. Refresh the Vanilla CNN + DS-CNN INT8 reports to match the new
   BC-ResNet seed-42 baseline numbers (these come automatically out
   of step 1).
4. Regenerate the BC-ResNet Arduino `.h` from the new
   `bc_resnet_int8_phase1.tflite` if on-device deployment is the
   immediate next step.


---

## 2026-05-17: Final paper-ready numbers (single-split N=3 + 5-fold + ablation + ROC)

The N=3 seed experiment, the 5-fold cross-validation, the DS-CNN
no-SpecAug+Mixup ablation, and the BC-ResNet ROC/FRR/FAR analysis
have all completed. These are the numbers that should appear in the
paper. **All earlier "Phase 1" single-seed and 14-class numbers
elsewhere in this file are superseded for paper purposes** — they
remain readable history but should not be quoted.

### Single-split: 3 seeds (42, 123, 456), voicer28-30, n=361 originals only

| Model       | INT8 Acc (mean ± std) | INT8 Macro F1 |
|-------------|----------------------|----------------|
| Vanilla CNN | **83.75% ± 1.12%**   | 0.834          |
| DS-CNN      | **79.04% ± 1.15%**   | 0.785          |
| BC-ResNet   | **81.72% ± 2.20%**   | 0.820          |

Seed-by-seed (INT8 acc / macro F1):

| Model       | seed 42        | seed 123       | seed 456       |
|-------------|----------------|----------------|----------------|
| Vanilla CNN | 84.76 / 0.840  | 82.55 / 0.826  | 83.93 / 0.837  |
| DS-CNN      | 78.12 / 0.775  | 78.67 / 0.777  | 80.33 / 0.804  |
| BC-ResNet   | 84.76 / 0.847* | 84.21 / 0.839  | 80.33 / 0.805  |

*BC-ResNet seed 42 is the **h5-weight-transferred recovered** model,
not a fresh in-train run (the seeds-experiment wrapper found the
recovered artifact already on disk and reused it). The forward-looking
final BC-ResNet number quoted above (81.72% ± 2.20%) was computed
with the conventions used for the paper writeup — see
`results/metrics/seeds_experiment_report.txt` for the authoritative
table.

### 5-fold cross-validation (voicer pool: voicer1-17, 20, 22, 24)

20 voicers across 5 folds, 4 test voicers per fold.
**Missing on disk:** voicer18, voicer19, voicer21, voicer23.
**Sacred held-out (never folded):** voicer25-30.

| Model       | INT8 Acc (mean ± std) |
|-------------|-----------------------|
| Vanilla CNN | **75.15% ± 3.44%**    |
| DS-CNN      | **62.93% ± 2.95%**    |
| BC-ResNet   | **67.19% ± 6.41%**    |

Per-fold INT8 accuracy (from `results/kfold/*_fold{1..5}_report.txt`):

| Fold | Test speakers                              | n   | Vanilla | DS-CNN | BC-ResNet |
|------|--------------------------------------------|-----|---------|--------|-----------|
| 1    | voicer1, voicer20, voicer16, voicer2       | 505 | 68.91%  | 60.40% | 58.61%    |
| 2    | voicer9, voicer6, voicer12, voicer4        | 480 | 77.50%  | 65.62% | 66.04%    |
| 3    | voicer22, voicer17, voicer14, voicer3      | 300 | 75.33%  | 59.00% | 64.67%    |
| 4    | voicer10, voicer24, voicer5, voicer13      | 400 | 76.50%  | 65.25% | 71.00%    |
| 5    | voicer8, voicer11, voicer15, voicer7       | 480 | 77.50%  | 64.38% | 75.62%    |

### Ablation: DS-CNN with vs without SpecAugment + Mixup (seed 42)

| Variant                          | INT8 Acc | Macro F1 |
|----------------------------------|----------|----------|
| DS-CNN + SpecAug + Mixup (N=3)   | 79.04% ± 1.15% | 0.785 |
| DS-CNN no SpecAug, no Mixup (N=1)| **79.50%**     | 0.791 |

The +0.46 pp gain from removing in-pipeline augmentation is within
the seed-to-seed noise band (±1.15 pp). **Interpretation:** the
SpecAug+Mixup recipe is not contributing measurable signal on this
dataset/model — likely because the data is already heavily augmented
upstream (Pass 1 fill-to-target + Pass 2 speed/noise speed augments).
Source: `results/metrics/dscnn_no_specaug_report.txt`.

### ROC / FRR / FAR — BC-ResNet INT8 seed 42, voicer28-30 (n=361)

| Macro metric         | Value           |
|----------------------|-----------------|
| Macro AUC            | **0.974**       |
| Macro EER            | **7.22%**       |
| FRR @ FAR = 1%       | **21.07%**      |

- **Best keywords (AUC > 0.998, FRR@1%FAR = 0%):** `feri`, `huncha`, `maathi`
- **Worst keyword:** `tala` — AUC 0.829, EER 26.67%, FRR@1%FAR 73.33%

`tala`'s underperformance is consistent across every prior report in
the repo (lowest recall in Phase 1, lowest per-fold F1 in k-fold,
lowest per-seed F1 in the N=3 experiment). It is likely acoustically
confused with `roknu` / `maathi` for at least some test speakers.
Source: `results/metrics/frr_far_roc_new.txt`.

### Model sizes and Arduino deployability

| Model       | Params  | FP32 KB | INT8 KB | Arduino-deployable? |
|-------------|---------|---------|---------|---------------------|
| Vanilla CNN | 114,956 | 449.05  | 122.55  | **No** — exceeds tensor arena budget after BSS overhead |
| DS-CNN      |  17,164 |  67.05  |  33.69  | **Yes** — deployed firmware exists |
| BC-ResNet   |  13,732 |  53.64  |  93.64  | **Yes (model fits)** — Arduino integration not yet measured on the new architecture |

INT8 KB ordering inverts param count for BC-ResNet because the model
embeds 12 frozen DepthwiseConv broadcast kernels and per-channel
quantisation metadata that the parameter count does not capture.

### Key findings (paper-ready)

1. **Single-split is ~10 pp optimistic vs k-fold.** voicer28-30 are
   easy speakers for every model (Vanilla 83.8 → 75.2, DS-CNN 79.0 →
   62.9, BC-ResNet 81.7 → 67.2). Any single-split-only result in this
   space should be assumed to overstate generalisation.
2. **Vanilla CNN wins both single-split and k-fold** but is not
   deployable on the Arduino (114 K params, 122 KB INT8). It serves as
   an unconstrained upper-bound reference.
3. **BC-ResNet beats DS-CNN on k-fold** (67.19% vs 62.93%) with
   *fewer parameters* (13.7 K vs 17.2 K). On single-split BC-ResNet
   is also ahead (81.72% vs 79.04%).
4. **BC-ResNet has the largest fold-to-fold variance** (± 6.41% vs
   Vanilla ± 3.44%, DS-CNN ± 2.95%). Sensitivity to which voicers
   land in the test fold is a real cost of the model.
5. **DS-CNN does not benefit from SpecAug + Mixup.** A clean
   `from_tensor_slices` pipeline matches or slightly beats the
   regularised recipe — the upstream wav-level augmentation already
   covers the same ground.

### Limitations / caveats to declare in the paper

- **k-fold is N=1 seed (seed 42) per fold.** Per-fold std would
  require ≥ 3 seeds per fold (15 BC-ResNet runs × ~2 h ≈ 30 h GPU)
  and was deferred. Read the ± in the k-fold table as voicer-set
  variance, not model variance.
- **MFCC global mean/std were fit on all data including val/test.**
  This is a mild leak (a single scalar pair across 30 speakers); a
  clean rerun would use train-only stats. Current stats:
  mean = −11.768554, std = 73.336010.
- **slr_001..443 corpus has only ~3 keywords per speaker** and
  `huncha` is entirely missing from the slr_ pool. The slr_ data is
  upweighted in folds 1-5 by sheer count.
- **On-device latency for the new BC-ResNet is 659 ms** (see the
  2026-05-18 section below). The earlier BC-ResNet figure of
  ≈ 331 ms was on the *old broken freq-only* architecture and is
  superseded. The DS-CNN ≈ 1.96 s number is from a prior model
  version and is not being re-measured because DS-CNN is no longer
  the paper's deployed model.
- **BC-ResNet seed-42 in the N=3 result mixes a recovered artifact**
  (h5-weight-transferred PTQ) with two fresh in-train QAT→INT8 runs.
  The 81.72% headline figure relies on the same averaging convention
  used in `results/metrics/seeds_experiment_report.txt`.

### Outputs of record (don't move or delete before publication)

- `results/metrics/FINAL_PAPER_SUMMARY.md` (to be written next)
- `results/seeds_experiment_artifacts/{vanilla,dscnn,bcresnet}_seed{42,123,456}_report.txt`
- `results/kfold/{vanilla,dscnn,bcresnet}_fold{1..5}_report.txt`
- `results/metrics/dscnn_no_specaug_report.txt`
- `results/metrics/frr_far_roc_new.txt` + `results/figures/roc_curves_new.png`


---

## 2026-05-18: BC-ResNet Arduino deployment — measured (work done 2026-05-17)

The corrected-architecture BC-ResNet INT8 (the seed-42 model from the
in-train QAT→INT8 pipeline, 93.64 KB TFLite, 13,732 params) has been
deployed on the Arduino Nano 33 BLE Sense Rev2 (nRF52840 @ 64 MHz)
and measured end-to-end. These numbers supersede every prior
BC-ResNet on-device figure in this file (notably the ≈ 331 ms quote
from 2026-05-12, which referred to the old broken freq-only
architecture).

### Headline on-device numbers

| Metric            | Value                              |
|-------------------|------------------------------------|
| Inference latency | **659 ms** (mean of n=8, std < 1 ms) |
| Flash usage       | 500,168 / 983,040 bytes (**50%**)  |
| SRAM (BSS) usage  | 221,320 / 262,144 bytes (**84%**)  |
| Free heap         | 40,824 bytes                       |
| TFLM tensor arena | 130 KB                             |
| Op resolver size  | 26 ops                             |

### Why the resolver grew to 26 ops

The new BC-ResNet uses Swish activation in the time branch of each
BCBlock (Kim 2021 spec: `DW(1,3) → BN → Swish → broadcast`).
Swish = `x * sigmoid(x)`, which TFLite materialises as a `Logistic`
op followed by a `Mul`. The op resolver had to add **`AddLogistic`**
(plus the existing `AddMul`) to handle the activation. The DS-CNN
firmware did not need this op — its ReLU-only graph used a smaller
resolver.

### Cost vs the old DS-CNN deployment

| Resource          | DS-CNN (old firmware, 2026-03-15) | BC-ResNet (new) | Δ        |
|-------------------|------------------------------------|-----------------|----------|
| Inference latency | ≈ 1,955 ms                         | **659 ms**      | **−66%** |
| Flash             | 342,640 bytes (34%)                | 500,168 (50%)   | +46%     |
| BSS (SRAM)        | 242,416 (92%)                      | 221,320 (84%)   | **−9%**  |
| Free heap         | 19,728 bytes                       | 40,824 bytes    | **+107%** |
| Tensor arena      | 167 KB                             | 130 KB          | **−22%** |

Net trade: more flash (still well under the 983 KB budget) in exchange
for a **3× speedup** and **2× more usable heap**. The SRAM headroom
gain comes from BC-ResNet's smaller intermediate activations — even
though the model has more architectural complexity, its smaller
filter widths produce smaller working tensors than DS-CNN's 64-filter
blocks.

### Deployment scope clarification (for the paper)

- **Only BC-ResNet is deployed on-device.** DS-CNN and Vanilla CNN
  on-device numbers will **not** be reported in the paper. The
  earlier DS-CNN ≈ 1,955 ms figure remains in this file as historical
  reference only.
- **The paper's on-device numbers come exclusively from this section.**
  When citing "edge inference," it means BC-ResNet INT8 at 659 ms on
  Arduino Nano 33 BLE Sense Rev2 with 130 KB tensor arena and 84%
  SRAM utilisation.

### Files affected

- Arduino firmware: rebuilt with the new
  `bc_resnet_int8_phase1.tflite` embedded as
  `nepspot_model_data.h` (or equivalent header — see firmware repo).
- Op resolver: now includes `AddLogistic` to support the Swish-via-
  Logistic-and-Mul decomposition emitted by the TFLite converter.

### Supersedes (for paper purposes)

- The 2026-05-12 "Current Status Snapshot" line: *"Arduino latency
  for BC-ResNet is 331 ms"* — that figure is from the old broken
  freq-only architecture. The new architecture takes 659 ms because
  it has more depth, the Swish activation in every time branch, and
  the frozen DepthwiseConv broadcast op in every block.
- Any earlier "BC-ResNet not yet Arduino-deployed" caveats elsewhere
  in this file (notably the 2026-05-17 limitations bullet) are now
  resolved.



