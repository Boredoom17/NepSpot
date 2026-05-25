"""Generate a stratified random sample of OpenSLR-mined clips for manual audit.

We extracted 913 keyword clips from OpenSLR-54 (Nepali corpus) using Whisper
word-position estimation. A P0 reviewer flag: no human verified that each
mined clip actually contains the intended keyword (folder name). This script
selects a reproducible 50-clip stratified sample so the audit can be done
manually.

What this script does:
  1. Enumerates all true originals under data/processed/slr_NNN/<keyword>/
     (basename matches the hex-id pattern produced by the mining script;
     augmented variants with `_aug_` infix or `aug_NN_` prefix are excluded).
  2. Picks ~50 clips, stratified across the 12 target keywords. Three
     keywords (huncha, thik_chha, baalnu) have zero mined clips and are
     simply absent from the sample.
  3. Resolves each clip's source wav under data/external/slr_db/speaker_NNN/
     (the slr_NNN <-> speaker_NNN folder rename is the only mapping needed).
  4. Writes the sample to results/metrics/openslr_audit_sample.csv with
     audit columns left blank for the human reviewer to fill in.

What this script does NOT do:
  - Listen to anything. Auditor instructions are printed at the end.

Run:
    cd /Users/ad/codes/NepSpot
    /Users/ad/codes/.venv310/bin/python3 scripts/audit_openslr_mining.py
"""

import csv
import os
import random
import re
import sys

SEED = 42

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
SLR_DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'slr_db')
CSV_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'openslr_audit_sample.csv')

KEYWORDS = [
    'aghillo', 'arko', 'baalnu', 'banda', 'feri', 'hoina',
    'huncha', 'maathi', 'roknu', 'suru', 'tala', 'thik_chha',
]

TARGET_TOTAL = 50

# Mining pipeline produces hex-id basenames (e.g. 6093638558.npy). Augmented
# files have either an `_aug_<kind>` suffix (Pass-2 speed augmentation) or an
# `aug_NN_<kind>` prefix (Pass-1 fill-to-target). Both have non-hex tokens.
HEX_ID_RE = re.compile(r'^[0-9a-f]+$')


def enumerate_originals():
    """Return dict[keyword] -> list[(slr_folder, clip_id, npy_path)]."""
    by_keyword = {kw: [] for kw in KEYWORDS}
    if not os.path.isdir(PROCESSED_DIR):
        raise FileNotFoundError('Missing processed dir: ' + PROCESSED_DIR)

    for entry in sorted(os.listdir(PROCESSED_DIR)):
        if not entry.startswith('slr_'):
            continue
        slr_dir = os.path.join(PROCESSED_DIR, entry)
        if not os.path.isdir(slr_dir):
            continue
        for keyword in sorted(os.listdir(slr_dir)):
            kw_dir = os.path.join(slr_dir, keyword)
            if not os.path.isdir(kw_dir) or keyword not in by_keyword:
                continue
            for fname in sorted(os.listdir(kw_dir)):
                if not fname.endswith('.npy'):
                    continue
                stem = fname[:-4]
                if not HEX_ID_RE.match(stem):
                    continue
                by_keyword[keyword].append((entry, stem, os.path.join(kw_dir, fname)))
    return by_keyword


def allocate_per_keyword(by_keyword, target_total):
    """Stratified allocation: floor(target/k) each, remainder to deepest pools."""
    mined = [kw for kw in KEYWORDS if len(by_keyword[kw]) > 0]
    if not mined:
        return {kw: 0 for kw in KEYWORDS}

    base = target_total // len(mined)
    remainder = target_total - base * len(mined)
    alloc = {kw: 0 for kw in KEYWORDS}
    for kw in mined:
        alloc[kw] = min(base, len(by_keyword[kw]))

    # Distribute remainder one slot per keyword max, to keep the per-keyword
    # count tight around ~base. Prefer keywords with the largest pools so the
    # extras land where the available data is deepest.
    ordered = sorted(mined, key=lambda k: (len(by_keyword[k]), k), reverse=True)
    for kw in ordered:
        if remainder == 0:
            break
        if alloc[kw] < len(by_keyword[kw]):
            alloc[kw] += 1
            remainder -= 1
    return alloc


def resolve_wav(slr_folder, keyword, clip_id):
    """slr_NNN -> speaker_NNN under data/external/slr_db/."""
    speaker_folder = slr_folder.replace('slr_', 'speaker_', 1)
    return os.path.join(SLR_DB_DIR, speaker_folder, keyword, clip_id + '.wav')


def main():
    rng = random.Random(SEED)

    by_keyword = enumerate_originals()
    total_originals = sum(len(v) for v in by_keyword.values())
    print('Found ' + str(total_originals) + ' original mined clips across '
          + str(sum(1 for v in by_keyword.values() if v)) + ' keywords.')
    for kw in KEYWORDS:
        print('  ' + kw.ljust(10) + ' : ' + str(len(by_keyword[kw])))

    alloc = allocate_per_keyword(by_keyword, TARGET_TOTAL)

    selected = []
    for kw in KEYWORDS:
        pool = by_keyword[kw]
        k = alloc[kw]
        if k == 0:
            continue
        # Sort for determinism, then sample reproducibly under SEED.
        pool_sorted = sorted(pool, key=lambda t: (t[0], t[1]))
        picks = rng.sample(pool_sorted, k)
        for slr_folder, clip_id, npy_path in picks:
            selected.append({
                'clip_id': clip_id,
                'intended_keyword': kw,
                'slr_folder': slr_folder,
                'npy_path': os.path.relpath(npy_path, PROJECT_ROOT),
                'wav_path': resolve_wav(slr_folder, kw, clip_id),
            })

    missing_wavs = [s for s in selected if not os.path.exists(s['wav_path'])]

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'clip_id', 'intended_keyword', 'wav_path',
            'keyword_present', 'well_cropped', 'notes',
        ])
        for s in selected:
            writer.writerow([
                s['clip_id'], s['intended_keyword'], s['wav_path'],
                '', '', '',
            ])

    print('')
    print('Stratified sample: ' + str(len(selected)) + ' clips')
    for kw in KEYWORDS:
        if alloc[kw] > 0:
            print('  ' + kw.ljust(10) + ' : ' + str(alloc[kw])
                  + '  (of ' + str(len(by_keyword[kw])) + ' available)')
    print('CSV: ' + os.path.relpath(CSV_PATH, PROJECT_ROOT))
    if missing_wavs:
        print('')
        print('WARNING: ' + str(len(missing_wavs))
              + ' selected clip(s) have no matching .wav under data/external/slr_db/. '
              + 'These rows are still in the CSV; the auditor will need to skip them or '
              + 'cross-reference data/raw/<slr_folder>/<keyword>/<clip_id>.wav.')
        for s in missing_wavs[:5]:
            print('  - ' + s['wav_path'])

    print('')
    print('Audit complete. To perform the audit:')
    print(' - Open the CSV file')
    print(' - For each clip, play the .wav file (use afplay <path> on macOS terminal)')
    print(' - Listen and fill in:')
    print('     keyword_present: yes / no / unclear')
    print('     well_cropped: yes (keyword starts within 0.5s and ends well before clip end) /')
    print('                    no (keyword is cut off, at the very edge, or missing) /')
    print('                    unclear')
    print('     notes: any observations (background speech, noise, etc.)')


if __name__ == '__main__':
    main()
