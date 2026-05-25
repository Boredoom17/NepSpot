#!/usr/bin/env python3
"""Augment only slr_* speakers with _aug_fast and _aug_fast_noisy.

This uses the existing `augment_for_speed` function from
`src/augment/run_augmentation.py` to preserve behavior.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'src' / 'augment'))

from run_augmentation import augment_for_speed, KEYWORDS, RAW_DIR

def main():
    raw_dir = Path(RAW_DIR)
    speakers = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('slr_')])
    total_added = 0
    print(f"Found {len(speakers)} slr_* speakers to augment")
    for sp in speakers:
        for kw in KEYWORDS:
            folder = sp / kw
            if not folder.exists():
                continue
            added = augment_for_speed(sp.name, kw, str(folder))
            total_added += added
        print(f"  OK {sp.name}")

    print(f"\nDone. Added {total_added} augmentation files (fast + noisy pairs)")

if __name__ == '__main__':
    main()
