# Data

The voice corpus is held privately. Audio data may be released separately on Hugging Face at a later date. This repository contains code, configs, trained models, and firmware only.

## What is tracked here

- `external/manifests/` — provenance manifests (accepted/rejected clip lists, source registry, speaker registry).
- `external/opensir/scan_keywords.py`, `external/opensir/utt_spk_text.tsv` — OpenSLR keyword-scanning helpers used to mine in-domain clips.

## What is intentionally NOT tracked

- `raw/` — raw 16 kHz WAV clips per speaker (gitignored).
- `processed/` — MFCC `.npy` feature caches (regenerable from raw audio via `src/features/extract_mfcc.py`).
- `external/staging/`, `external/openslr/clips/`, `external/commonvoice/clips/` — downloaded source corpora.
