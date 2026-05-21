# Option 4 Complete Workflow (Automated)

## What I just did for you

1. **Downloaded** Common Voice Nepali (39 MB) ✓
2. **Extracted** the archive ✓
3. **Scanned** all transcripts for your 12 keywords ✓
4. **Found** 119 matching clips ✓
5. **Converted** all MP3s to 16kHz mono WAV ✓
6. **Logged** everything in candidates.csv ✓

## What you do now

### Option A: Quick Review (recommended first)

1. Open terminal in NepSpot root.
2. Run:
   ```bash
   python3 scripts/data_mining/03_review_clips.py
   ```
3. For each clip:
   - You'll hear the audio
   - Type: `y` (accept), `n` (reject), `s` (skip), or `q` (quit)
4. It updates accepted.csv and rejected.csv automatically.

### Option B: Skip Review (not recommended)

If you trust the automatic filtering, skip clips and accept all huncha, feri, suru (the common ones).

## Results so far

From 119 Common Voice clips found:
- huncha: 77 clips
- feri: 13 clips
- suru: 8 clips
- tala: 7 clips
- maathi: 4 clips
- hoina: 2 clips
- banda: 2 clips
- thik_chha: 6 clips
- **Not found**: aghillo, roknu, arko (rare words - we'll need YouTube for these)

## After review

Once you accept clips:

1. Copy to NepSpot train format:
   ```bash
   python3 scripts/data_mining/04_copy_to_nepspot.py
   ```

2. Extract MFCC features:
   ```bash
   python src/features/extract_mfcc.py
   ```

3. Retrain models:
   ```bash
   python src/models/train.py
   python src/models/train_vanilla.py
   python src/models/train_bcresnet.py
   ```

Then repeat for OpenSLR + YouTube if needed.
