#!/usr/bin/env python3
import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import importlib.util

run_aug_path = os.path.join(ROOT, 'src', 'utils', 'run_augmentation.py')
spec = importlib.util.spec_from_file_location('run_augmentation', run_aug_path)
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)

CONFIG_PATH = os.path.join(ROOT, 'configs', 'speaker_split_v1.json')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')

def load_splits():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as handle:
        cfg = json.load(handle)
    splits = cfg.get('splits', {})
    train = list(splits.get('train_complete', [])) + list(splits.get('train_partial', []))
    return train

def find_external_train_speakers():
    train = load_splits()
    if not os.path.exists(RAW_DIR):
        return []
    speakers = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    # external speakers are those starting with 'speaker_'
    external = sorted([s for s in speakers if s.startswith('speaker_') and s in train])
    return external

def count_wavs_in_speaker(speaker):
    total = 0
    speaker_dir = os.path.join(RAW_DIR, speaker)
    for word in ru.KEYWORDS:
        folder = os.path.join(speaker_dir, word)
        if not os.path.exists(folder):
            continue
        total += len([f for f in os.listdir(folder) if f.endswith('.wav')])
    return total


def normalize_speaker_structure(speaker):
    """Move wav files directly under speaker dir into keyword subfolders when possible."""
    speaker_dir = os.path.join(RAW_DIR, speaker)
    moved = 0
    for fname in os.listdir(speaker_dir):
        path = os.path.join(speaker_dir, fname)
        if not os.path.isfile(path) or not fname.endswith('.wav'):
            continue
        lower = fname.lower()
        matched = None
        for kw in ru.KEYWORDS:
            if kw in lower:
                matched = kw
                break
        if matched:
            target_dir = os.path.join(speaker_dir, matched)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, fname)
            try:
                os.rename(path, new_path)
                moved += 1
            except Exception:
                pass
    return moved

def main():
    speakers = find_external_train_speakers()
    if not speakers:
        print('No external train speakers found to augment.')
        return

    print(f'Found {len(speakers)} external train speakers:')
    print(', '.join(speakers))

    total_added = 0
    before_counts = {s: count_wavs_in_speaker(s) for s in speakers}
    print('\nCounts before augmentation (per-speaker total WAVs):')
    for s, c in before_counts.items():
        print(f'  {s}: {c}')

    # Run augmentation per-speaker
    for s in speakers:
        # normalize structure (move wavs into keyword folders where filenames contain keywords)
        moved = normalize_speaker_structure(s)
        if moved:
            print(f'  moved {moved} wavs into per-keyword folders for {s}')
        print(f'\nAugmenting: {s}')
        speaker_dir = os.path.join(RAW_DIR, s)
        for word in ru.KEYWORDS:
            folder = os.path.join(speaker_dir, word)
            if not os.path.exists(folder):
                continue
            added1 = ru.fill_to_target(s, word, folder)
            if added1:
                print(f'  filled {word}: +{added1}')
            added2 = ru.augment_for_speed(s, word, folder)
            if added2:
                print(f'  speed aug {word}: +{added2}')
            total_added += (added1 + added2)

    after_counts = {s: count_wavs_in_speaker(s) for s in speakers}
    print('\nCounts after augmentation (per-speaker total WAVs):')
    for s in speakers:
        print(f'  {s}: {before_counts.get(s,0)} -> {after_counts.get(s,0)}')

    avg_before = sum(before_counts.values()) / len(speakers)
    avg_after  = sum(after_counts.values()) / len(speakers)
    print(f'\nAverage per-external-speaker: before={avg_before:.1f}, after={avg_after:.1f}')
    print(f'Total augmented clips added: {total_added}')

if __name__ == '__main__':
    main()
