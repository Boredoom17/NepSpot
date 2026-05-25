#!/usr/bin/env python3
"""Organize CommonVoice clips into data/raw/cv_NNN/<keyword>/filename.wav.

This script copies CV clips from staging into data/raw with a deterministic
mapping cv_001..cv_NNN based on sorted speaker hashes.
"""
import shutil
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'external' / 'staging' / 'raw_clips'
DST_ROOT = ROOT / 'data' / 'raw'


def find_cv_hashes():
    hashes = set()
    for p in SRC.glob('cv_*.wav'):
        parts = p.name.split('_')
        if len(parts) >= 3:
            hashes.add(parts[1])
    return sorted(hashes)


def organize(copy=True):
    hashes = find_cv_hashes()
    mapping = {h: f'cv_{i:03d}' for i, h in enumerate(hashes, start=1)}
    for p in SRC.glob('cv_*.wav'):
        parts = p.name.split('_')
        h = parts[1]
        kw = parts[-1].replace('.wav', '')
        dst_dir = DST_ROOT / mapping[h] / kw
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / p.name
        if not dst_file.exists():
            if copy:
                shutil.copy2(p, dst_file)
            else:
                os.link(p, dst_file)
    print('Organized', len(hashes), 'CV speakers under', DST_ROOT)


if __name__ == '__main__':
    organize(copy=True)
