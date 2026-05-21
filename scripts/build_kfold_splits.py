"""Build deterministic 5-fold speaker-independent split configs for NepSpot.

K-fold pool = the 20 train-side voicers actually present under
data/processed/. voicer18, voicer19, voicer21, voicer23 are MISSING from
data/processed/ and excluded from the pool. voicer25..voicer30 are also
excluded — those are the held-out final-test speakers and must never be
touched by k-fold.

20 voicers split into 5 folds of size 4 each via a seed=42 shuffle.
Val rotates with the test window: each fold's val is the NEXT 2 voicers
after its test block in the shuffled order, wrapping around at the end.
That guarantees zero test/val overlap within a fold AND that no voicer
appears in val more than once across the 5 folds.

For each fold (size 4 test / 2 val):
  - TEST  = positions pos..pos+3 in the shuffled order
  - VAL   = the next 2 positions, wrapping around
  - TRAIN = the remaining 14 voicers + every slr_* and speaker_*
            folder found under data/processed/

Writes one JSON file per fold at configs/kfold/fold_{N}.json so the k-fold
trainer can pick them up without re-deriving the shuffle. Pure setup —
this script does not load MFCCs and does not train anything.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SEED = 42
N_FOLDS = 5
VAL_PER_FOLD = 2

# Only voicers that actually have folders under data/processed/.
# voicer18, voicer19, voicer21, voicer23 are missing on disk.
# voicer25..voicer30 are the held-out final-test split — never touched here.
KFOLD_POOL = [
    'voicer1',  'voicer2',  'voicer3',  'voicer4',  'voicer5',
    'voicer6',  'voicer7',  'voicer8',  'voicer9',  'voicer10',
    'voicer11', 'voicer12', 'voicer13', 'voicer14', 'voicer15',
    'voicer16', 'voicer17', 'voicer20', 'voicer22', 'voicer24',
]
MISSING_VOICERS = ['voicer18', 'voicer19', 'voicer21', 'voicer23']
HELD_OUT_VOICERS = [f'voicer{i}' for i in range(25, 31)]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
CONFIG_DIR = PROJECT_ROOT / 'configs' / 'kfold'


def enumerate_extra_speakers():
    """Find slr_* and speaker_* folders actually present under data/processed/."""
    if not PROCESSED_DIR.is_dir():
        raise FileNotFoundError(f'Missing processed data dir: {PROCESSED_DIR}')

    def _sort_key(name):
        # Sort slr_001..slr_443 / speaker_1..speaker_23 numerically.
        tail = name.split('_', 1)[1] if '_' in name else name
        try:
            return (0, int(tail))
        except ValueError:
            return (1, tail)

    slr = sorted(
        (p.name for p in PROCESSED_DIR.iterdir() if p.is_dir() and p.name.startswith('slr_')),
        key=_sort_key,
    )
    speaker_extra = sorted(
        (p.name for p in PROCESSED_DIR.iterdir() if p.is_dir() and p.name.startswith('speaker_')),
        key=_sort_key,
    )
    return slr, speaker_extra


def compute_folds():
    rng = np.random.RandomState(SEED)
    shuffled = list(rng.permutation(KFOLD_POOL))
    n = len(shuffled)
    base, rem = divmod(n, N_FOLDS)  # 20 / 5 -> 4 r 0 -> sizes [4,4,4,4,4]
    sizes = [base + 1 if i < rem else base for i in range(N_FOLDS)]

    folds = []
    pos = 0
    for fold_idx, size in enumerate(sizes):
        test_speakers = shuffled[pos:pos + size]
        # Val: the next VAL_PER_FOLD positions after test, wrapping around.
        val_start = (pos + size) % n
        val_speakers = [shuffled[(val_start + i) % n] for i in range(VAL_PER_FOLD)]
        pos += size

        test_set = set(test_speakers)
        val_set = set(val_speakers)
        train_voicers = [v for v in shuffled if v not in test_set and v not in val_set]

        folds.append({
            'fold': fold_idx + 1,
            'test_speakers': test_speakers,
            'val_speakers': val_speakers,
            'train_voicers': train_voicers,
        })
    return folds, shuffled


def write_fold_configs(folds, slr, speaker_extra):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for f in folds:
        path = CONFIG_DIR / f'fold_{f["fold"]}.json'
        payload = {
            'fold': f['fold'],
            'n_folds': N_FOLDS,
            'seed': SEED,
            'kfold_pool': KFOLD_POOL,
            'missing_on_disk_excluded': MISSING_VOICERS,
            'held_out_speakers': HELD_OUT_VOICERS,
            'test_speakers': f['test_speakers'],
            'val_speakers': f['val_speakers'],
            'train_voicers': f['train_voicers'],
            'train_slr': slr,
            'train_speakers_extra': speaker_extra,
            'val_aug_policy': 'exclude (_aug_*.npy filtered)',
            'test_aug_policy': 'exclude (_aug_*.npy filtered)',
            'train_aug_policy': 'include all .npy files',
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        written.append(path)
    return written


def main():
    # Disk-presence guard: every voicer in KFOLD_POOL must have a folder
    # under data/processed/. Any voicer that doesn't exist on disk would
    # silently contribute 0 samples to its fold's test/val and is a bug.
    missing = [v for v in KFOLD_POOL if not (PROCESSED_DIR / v).is_dir()]
    if missing:
        raise FileNotFoundError(
            f'KFOLD_POOL contains voicers without folders under '
            f'{PROCESSED_DIR}: {missing}'
        )

    folds, shuffled = compute_folds()
    slr, speaker_extra = enumerate_extra_speakers()

    print('=' * 78)
    print('K-Fold split build — speaker-independent, seed=42')
    print('=' * 78)
    print(f'K-fold pool ({len(KFOLD_POOL)}): {KFOLD_POOL}')
    print(f'Shuffled order:    {shuffled}')
    print(f'Excluded — missing on disk: {MISSING_VOICERS}')
    print(f'Held-out (NEVER touched):   {", ".join(HELD_OUT_VOICERS)}')
    print()
    print(f'Extra train speakers detected under data/processed/:')
    print(f'  slr_*     : {len(slr)} folders'
          + (f'  (range {slr[0]} .. {slr[-1]})' if slr else ''))
    print(f'  speaker_* : {len(speaker_extra)} folders'
          + (f'  (range {speaker_extra[0]} .. {speaker_extra[-1]})' if speaker_extra else ''))
    print()

    fold_size_total = 0
    for f in folds:
        fold_size_total += len(f['test_speakers'])
        print(f'Fold {f["fold"]}/{N_FOLDS}:')
        print(f'  TEST  ({len(f["test_speakers"]):>2}): {", ".join(f["test_speakers"])}')
        print(f'  VAL   ({len(f["val_speakers"]):>2}): {", ".join(f["val_speakers"])}')
        print(f'  TRAIN ({len(f["train_voicers"]):>2} voicers): {", ".join(f["train_voicers"])}')
        print(f'        + {len(slr)} slr_* + {len(speaker_extra)} speaker_* '
              f'= {len(slr) + len(speaker_extra)} extra speakers')
        print()

    print(f'Sanity: union of all test folds covers {fold_size_total} voicers '
          f'(expected {len(KFOLD_POOL)})')
    assert fold_size_total == len(KFOLD_POOL), 'fold sizes do not sum to pool size'

    # 1. Each voicer in test exactly once across folds.
    all_test = []
    for f in folds:
        all_test.extend(f['test_speakers'])
    assert len(set(all_test)) == len(all_test), 'duplicate voicer across test folds'
    assert set(all_test) == set(KFOLD_POOL), 'test folds do not cover full pool'

    # 2. No voicer is in test AND val of the same fold.
    for f in folds:
        overlap = set(f['test_speakers']) & set(f['val_speakers'])
        assert not overlap, f'fold {f["fold"]} has test/val overlap: {overlap}'

    # 3. No voicer appears in val more than once across folds.
    all_val = []
    for f in folds:
        all_val.extend(f['val_speakers'])
    val_counts = {v: all_val.count(v) for v in set(all_val)}
    repeats = {v: c for v, c in val_counts.items() if c > 1}
    assert not repeats, f'voicers appearing in val more than once: {repeats}'

    # 4. Train pool per fold is exactly KFOLD_POOL \ (test ∪ val).
    for f in folds:
        used = set(f['test_speakers']) | set(f['val_speakers'])
        expected_train = [v for v in KFOLD_POOL if v not in used]
        assert sorted(f['train_voicers']) == sorted(expected_train), (
            f'fold {f["fold"]} train_voicers does not match pool minus test+val'
        )

    n_unique_val = len(set(all_val))
    print(f'Sanity: val voicers across folds are all distinct '
          f'({n_unique_val} unique = {N_FOLDS} folds × {VAL_PER_FOLD})')
    print(f'Sanity: every fold has zero test/val overlap')
    print(f'Sanity: {len(set(KFOLD_POOL) - set(all_val))} voicers '
          f'never appear in val (math: {len(KFOLD_POOL)} - {n_unique_val} = '
          f'{len(KFOLD_POOL) - n_unique_val})')

    written = write_fold_configs(folds, slr, speaker_extra)
    print('-' * 78)
    print(f'Wrote {len(written)} fold configs to {CONFIG_DIR.relative_to(PROJECT_ROOT)}/:')
    for p in written:
        print(f'  {p.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
