# Step-by-Step Beginner Guide (Option 4)

This guide assumes you are a beginner and want a realistic path.

## Phase 0: Setup (Do this once)

1. Open terminal in NepSpot root.
2. Run:

   bash scripts/data_mining/make_dataset_scaffold.sh

3. Confirm these folders exist:
   - data/external/incoming/commonvoice/
   - data/external/incoming/openslr/
   - data/external/incoming/youtube/
   - data/external/manifests/
   - data/external/staging/raw_clips/
   - data/external/staging/for_nepspot_raw/

4. Open and keep these files updated while working:
   - data/external/manifests/candidates.csv
   - data/external/manifests/accepted.csv
   - data/external/manifests/rejected.csv
   - data/external/manifests/speaker_registry.csv
   - data/external/manifests/source_registry.csv

## Phase 1: Common Voice (highest priority)

1. Download Nepali Common Voice dataset.
2. Place the raw download inside data/external/incoming/commonvoice/.
3. Parse transcript TSV/CSV and search for keyword matches using aliases from configs/data_mining/keyword_aliases.json.
4. For each match, add one row to candidates.csv.
5. Cut candidate clips around keyword timestamps into data/external/staging/raw_clips/.
6. Normalize clips to 16kHz, mono, around 1.0s.
7. Listen and review clips manually.
8. If good, add row to accepted.csv; if bad, add row to rejected.csv.

## Phase 2: OpenSLR (fill gaps)

1. Repeat Phase 1 workflow using data/external/incoming/openslr/.
2. Focus on keywords with low accepted counts.
3. Prefer adding new speakers over adding too many clips from one speaker.

## Phase 3: Targeted YouTube (sparse keywords only)

1. Create a shortlist of channels/videos where sparse keywords are likely.
2. Log each source in source_registry.csv before extracting clips.
3. Keep provenance fields complete (video URL, timestamp, keyword, speaker tag if possible).
4. Extract candidate clips and follow the same review process as Phase 1.
5. Do not plan to release raw YouTube audio publicly.

## Phase 4: Prepare for NepSpot training

1. Move accepted clips into NepSpot raw format:

   data/raw/<speaker_id>/<keyword>/<clip>.wav

2. Recommended speaker id prefixes:
   - cv_<id>
   - slr_<id>
   - yt_<id>

3. Keep all external speakers in train only. Do not add them to val/test split.

## Phase 5: Rebuild features and train

1. Re-extract MFCC:

   python src/features/extract_mfcc.py

2. Train models (same protocol as before):
   - python src/models/train.py
   - python src/models/train_vanilla.py
   - python src/models/train_bcresnet.py

3. Compare updated metrics and document source-wise speaker/clip stats in your report.

## Quality Checklist (must pass)

- Audio is clearly Nepali speech.
- Target keyword is actually present and centered.
- Clip is not duplicated.
- Duration and sample rate match project settings.
- Metadata row exists for every accepted clip.

## Minimum Result Target

- 100+ effective speakers total across in-house + external sources.
- Per-keyword balance so rare words are not underrepresented.
- Fair evaluation preserved by unchanged validation/test speakers.
