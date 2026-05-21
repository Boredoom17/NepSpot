#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import random
import re
import tarfile
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / 'src'
if str(SRC_DIR / 'features') not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR / 'features'))
if str(SRC_DIR / 'utils') not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR / 'utils'))

from extract_mfcc import extract_mfcc_raw, load_and_pad  # type: ignore
from augment import add_noise, load_audio, time_stretch  # type: ignore
from dataset import KEYWORDS, ensure_directory, load_split_config, summarize_split  # type: ignore

SAMPLE_RATE = 16000
TARGET_CLIPS = 500
SILENCE_THRESHOLD_STEPS = [0.005, 0.0075, 0.01]
SILENCE_WINDOWS_PER_SOURCE_LIMIT = 5
UNKNOWN_PER_SOURCE_LIMIT = 2
UNKNOWN_MIN_RMS = 0.05
RNG = random.Random(42)

CV_ARCHIVE = ROOT / 'data' / 'external' / 'incoming' / 'commonvoice' / '1774118379910-cv-corpus-25.0-2026-03-09-ne-NP.tar.gz'
CV_PREFIX = 'cv-corpus-25.0-2026-03-09/ne-NP'
CV_TSV_NAMES = ['train.tsv', 'validated.tsv', 'other.tsv']
OPENSLR_TSV = ROOT / 'data' / 'external' / 'opensir' / 'utt_spk_text.tsv'
OPENSLR_ZIPS_DIR = ROOT / 'data' / 'external' / 'opensir' / 'zips'
SLR_DB_DIR = ROOT / 'data' / 'external' / 'slr_db'
RAW_SILENCE_DIR = ROOT / 'data' / 'raw' / '_silence' / 'silence'
RAW_UNKNOWN_DIR = ROOT / 'data' / 'raw' / '_unknown' / 'unknown'
PROCESSED_SILENCE_DIR = ROOT / 'data' / 'processed' / '_silence' / 'silence'
PROCESSED_UNKNOWN_DIR = ROOT / 'data' / 'processed' / '_unknown' / 'unknown'
MFCC_MEAN = -12.608452796936035
MFCC_STD = 78.48593139648438

FORBIDDEN_SUBSTRINGS = [
    'अघिल्लो', 'अर्को', 'बाल्नु', 'बन्द', 'बन्दा',
    'फेरि', 'होइन', 'होइनन्', 'हुन्छ', 'हुन्छन्',
    'माथि', 'रोक्नु', 'सुरु', 'तल', 'ठिक छ',
]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text or '')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def contains_forbidden_keyword(text: str) -> bool:
    normalized = normalize_text(text)
    return any(fragment in normalized for fragment in FORBIDDEN_SUBSTRINGS)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def pad_or_trim(audio: np.ndarray, target_len: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len]
    if len(audio) < target_len:
        pad = target_len - len(audio)
        return np.pad(audio, (0, pad), mode='constant')
    return audio


def save_audio(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    ensure_parent(path)
    sf.write(str(path), audio.astype(np.float32), sr)


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def load_audio_from_bytes(data: bytes, suffix: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = Path(handle.name)
    try:
        try:
            audio, _ = librosa.load(str(temp_path), sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32)
        except Exception:
            segment = AudioSegment.from_file(str(temp_path))
            segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
            samples = np.array(segment.get_array_of_samples())
            if segment.sample_width == 1:
                audio = samples.astype(np.float32) / 128.0
            elif segment.sample_width == 2:
                audio = samples.astype(np.float32) / 32768.0
            elif segment.sample_width == 4:
                audio = samples.astype(np.float32) / 2147483648.0
            else:
                audio = samples.astype(np.float32)
            return audio
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_audio_from_path(path: Path) -> np.ndarray:
    audio, _ = load_audio(str(path), sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def load_audio_from_tar_member(tar: tarfile.TarFile, member_name: str) -> np.ndarray:
    member = tar.getmember(member_name)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member_name)
    data = extracted.read()
    return load_audio_from_bytes(data, Path(member_name).suffix)


def load_audio_from_zip_member(zip_path: Path, member_name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        data = zf.read(member_name)
    return load_audio_from_bytes(data, Path(member_name).suffix)


def current_original_clip_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.is_file() and f.suffix == '.wav' and '_aug_' not in f.name])


def current_total_clip_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir() if f.is_file() and f.suffix == '.wav'])


def clear_folder_files(folder: Path, suffix: str) -> None:
    if not folder.exists():
        return
    for path in folder.iterdir():
        if path.is_file() and path.suffix == suffix:
            path.unlink()


def parse_cv_tsv(tar: tarfile.TarFile, tsv_name: str) -> pd.DataFrame:
    member_name = f'{CV_PREFIX}/{tsv_name}'
    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    return pd.read_csv(extracted, sep='\t')


def build_cv_unknown_candidates() -> List[Dict]:
    candidates: List[Dict] = []
    if not CV_ARCHIVE.exists():
        return candidates

    with tarfile.open(CV_ARCHIVE, 'r:gz') as tar:
        for tsv_name in CV_TSV_NAMES:
            try:
                frame = parse_cv_tsv(tar, tsv_name)
            except Exception:
                continue

            if 'path' not in frame.columns or 'sentence' not in frame.columns:
                continue

            for _, row in frame.iterrows():
                sentence = str(row.get('sentence', '')).strip()
                clip_name = str(row.get('path', '')).strip()
                if not clip_name or contains_forbidden_keyword(sentence):
                    continue
                member_name = f'{CV_PREFIX}/clips/{clip_name}'
                try:
                    tar.getmember(member_name)
                except KeyError:
                    continue
                candidates.append({
                    'kind': 'cv_tar',
                    'source_key': clip_name,
                    'member': member_name,
                    'sentence': sentence,
                })
    return candidates


def build_openslr_member_index() -> Dict[str, Tuple[Path, str]]:
    member_index: Dict[str, Tuple[Path, str]] = {}
    if not OPENSLR_ZIPS_DIR.exists():
        return member_index

    for zip_path in sorted(OPENSLR_ZIPS_DIR.glob('*.zip')):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(('.flac', '.wav', '.mp3')):
                        continue
                    stem = Path(name).stem
                    if stem and stem not in member_index:
                        member_index[stem] = (zip_path, name)
        except Exception:
            continue
    return member_index


def build_openslr_unknown_candidates(member_index: Dict[str, Tuple[Path, str]]) -> List[Dict]:
    candidates: List[Dict] = []
    if not OPENSLR_TSV.exists():
        return candidates

    frame = pd.read_csv(OPENSLR_TSV, sep='\t', header=None, names=['file_id', 'speaker_id', 'text'])
    for _, row in frame.iterrows():
        text = str(row['text']).strip()
        if contains_forbidden_keyword(text):
            continue
        file_id = str(row['file_id']).strip()
        if file_id not in member_index:
            continue
        zip_path, member_name = member_index[file_id]
        candidates.append({
            'kind': 'zip',
            'source_key': file_id,
            'archive': zip_path,
            'member': member_name,
            'speaker_id': str(row['speaker_id']),
            'text': text,
        })
    return candidates


def build_silence_sources() -> List[Dict]:
    sources: List[Dict] = []

    if CV_ARCHIVE.exists():
        with tarfile.open(CV_ARCHIVE, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.startswith(f'{CV_PREFIX}/clips/') and member.name.endswith('.mp3'):
                    sources.append({
                        'kind': 'cv_tar',
                        'source_key': member.name,
                        'member': member.name,
                    })

    if SLR_DB_DIR.exists():
        for wav_path in sorted(SLR_DB_DIR.rglob('*.wav')):
            sources.append({
                'kind': 'file',
                'source_key': str(wav_path),
                'path': wav_path,
            })

    return sources


def load_source_audio(record: Dict) -> np.ndarray:
    kind = record['kind']
    if kind == 'file':
        return load_audio_from_path(record['path'])
    if kind == 'cv_tar':
        with tarfile.open(CV_ARCHIVE, 'r:gz') as tar:
            return load_audio_from_tar_member(tar, record['member'])
    if kind == 'zip':
        return load_audio_from_zip_member(record['archive'], record['member'])
    raise ValueError(f'Unsupported source kind: {kind}')


def pick_middle_segment(audio: np.ndarray, segment_len: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio) <= segment_len:
        return pad_or_trim(audio, segment_len)

    min_start = int(0.2 * len(audio))
    max_start = int(0.8 * len(audio)) - segment_len
    if max_start <= min_start:
        start = max(0, (len(audio) - segment_len) // 2)
    else:
        start = RNG.randint(min_start, max_start)
    return audio[start:start + segment_len]


def find_silence_windows(audio: np.ndarray, threshold: float) -> List[Tuple[int, np.ndarray]]:
    windows: List[Tuple[int, np.ndarray]] = []
    if len(audio) == 0:
        return windows

    padded = pad_or_trim(audio)
    if len(audio) <= SAMPLE_RATE:
        window_rms = rms(padded)
        if window_rms < threshold:
            windows.append((0, padded))
        return windows

    hop = SAMPLE_RATE // 2
    for start in range(0, len(audio) - SAMPLE_RATE + 1, hop):
        window = audio[start:start + SAMPLE_RATE]
        if rms(window) < threshold:
            windows.append((start, window.copy()))
    return windows


def mine_silence(target: int = TARGET_CLIPS) -> List[Path]:
    ensure_directory(str(RAW_SILENCE_DIR))
    existing = current_original_clip_count(RAW_SILENCE_DIR)
    next_index = existing + 1
    if existing >= target:
        print(f'Silence already has {existing} raw clips; skipping mining.')
        return sorted([p for p in RAW_SILENCE_DIR.glob('*.wav') if '_aug_' not in p.name])

    saved: List[Path] = []
    per_source_counts: Dict[str, int] = defaultdict(int)
    seen_windows = set()

    sources = build_silence_sources()
    RNG.shuffle(sources)

    print(f'Found {len(sources)} candidate silence source files')
    for threshold in SILENCE_THRESHOLD_STEPS:
        if next_index > target:
            break
        print(f'  Silence threshold pass: {threshold}')
        for record in sources:
            if next_index > target:
                break
            source_key = record['source_key']
            if per_source_counts[source_key] >= SILENCE_WINDOWS_PER_SOURCE_LIMIT:
                continue
            try:
                audio = load_source_audio(record)
            except Exception:
                continue
            windows = find_silence_windows(audio, threshold)
            for start, window in windows:
                if next_index > target:
                    break
                window_id = (source_key, start)
                if window_id in seen_windows:
                    continue
                seen_windows.add(window_id)
                window = pad_or_trim(window)
                out_path = RAW_SILENCE_DIR / f'clip_{next_index:04d}.wav'
                save_audio(out_path, window)
                saved.append(out_path)
                next_index += 1
                per_source_counts[source_key] += 1
                if per_source_counts[source_key] >= SILENCE_WINDOWS_PER_SOURCE_LIMIT:
                    break

    return saved


def mine_unknown(target: int = TARGET_CLIPS) -> List[Path]:
    ensure_directory(str(RAW_UNKNOWN_DIR))
    clear_folder_files(RAW_UNKNOWN_DIR, '.wav')
    clear_folder_files(PROCESSED_UNKNOWN_DIR, '.npy')
    existing = current_original_clip_count(RAW_UNKNOWN_DIR)
    next_index = existing + 1
    if existing >= target:
        print(f'Unknown already has {existing} raw clips; skipping mining.')
        return sorted([p for p in RAW_UNKNOWN_DIR.glob('*.wav') if '_aug_' not in p.name])

    member_index = build_openslr_member_index()
    cv_candidates = build_cv_unknown_candidates()
    slr_candidates = build_openslr_unknown_candidates(member_index)
    candidates = cv_candidates + slr_candidates
    RNG.shuffle(candidates)

    print(f'Found {len(cv_candidates)} Common Voice unknown candidates')
    print(f'Found {len(slr_candidates)} OpenSLR unknown candidates')

    saved: List[Path] = []
    per_source_counts: Dict[str, int] = defaultdict(int)

    for record in candidates:
        if next_index > target:
            break
        source_key = record['source_key']
        if per_source_counts[source_key] >= UNKNOWN_PER_SOURCE_LIMIT:
            continue
        try:
            audio = load_source_audio(record)
        except Exception:
            continue
        segment = pick_middle_segment(audio)
        segment = pad_or_trim(segment)
        if rms(segment) < UNKNOWN_MIN_RMS:
            continue
        out_path = RAW_UNKNOWN_DIR / f'clip_{next_index:04d}.wav'
        save_audio(out_path, segment)
        saved.append(out_path)
        next_index += 1
        per_source_counts[source_key] += 1

    return saved


def augment_folder(folder: Path) -> int:
    if not folder.exists():
        return 0

    added = 0
    real_clips = sorted([
        path for path in folder.iterdir()
        if path.is_file() and path.suffix == '.wav' and '_aug_' not in path.name
    ])

    for clip_path in real_clips:
        basename = clip_path.stem
        fast_path = folder / f'{basename}_aug_fast.wav'
        noisy_path = folder / f'{basename}_aug_fast_noisy.wav'
        if fast_path.exists() and noisy_path.exists():
            continue

        try:
            audio, sr = load_audio(str(clip_path), sr=SAMPLE_RATE)
            rate = RNG.uniform(1.2, 1.35)
            fast = time_stretch(audio, rate=rate)
            fast = pad_or_trim(np.asarray(fast, dtype=np.float32))
            if not fast_path.exists():
                sf.write(str(fast_path), fast.astype(np.float32), sr)
                added += 1
            noisy = add_noise(fast, noise_factor=0.005)
            noisy = pad_or_trim(np.asarray(noisy, dtype=np.float32))
            if not noisy_path.exists():
                sf.write(str(noisy_path), noisy.astype(np.float32), sr)
                added += 1
        except Exception as exc:
            print(f'  Augmentation failed for {clip_path.name}: {exc}')

    return added


def extract_new_features(raw_folder: Path, output_folder: Path, fixed_mean: float, fixed_std: float) -> List[Path]:
    ensure_directory(str(output_folder))
    generated: List[Path] = []

    wav_files = sorted([
        path for path in raw_folder.iterdir()
        if path.is_file() and path.suffix == '.wav'
    ])

    for wav_path in wav_files:
        out_path = output_folder / wav_path.with_suffix('.npy').name
        if out_path.exists():
            continue
        audio = load_and_pad(str(wav_path), sr=SAMPLE_RATE, duration=1.0)
        mfcc = extract_mfcc_raw(audio, sr=SAMPLE_RATE)
        normalized = (mfcc - fixed_mean) / (fixed_std + 1e-8)
        np.save(str(out_path), normalized.astype(np.float32))
        generated.append(out_path)

    return generated


def sample_rms_values(folder: Path, sample_count: int = 20) -> List[float]:
    files = sorted([path for path in folder.glob('*.wav') if path.is_file()])
    if not files:
        return []
    sample_count = min(sample_count, len(files))
    sample_files = RNG.sample(files, sample_count)
    values: List[float] = []
    for path in sample_files:
        audio = load_audio_from_path(path)
        values.append(rms(audio))
    return values


def count_feature_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len([path for path in folder.glob('*.npy') if path.is_file()])


def label_summary() -> List[str]:
    return list(sorted(set(KEYWORDS)))


def update_split_summary() -> Dict:
    return summarize_split('train')


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    config = load_split_config()
    train_speakers = list(config.get('splits', {}).get('train_complete', [])) + list(config.get('splits', {}).get('train_partial', []))
    if '_silence' not in train_speakers or '_unknown' not in train_speakers:
        print('Warning: split config has not been updated with _silence/_unknown yet.')

    print('=== Step 1: Silence mining ===')
    silence_saved = mine_silence(TARGET_CLIPS)
    print(f'Silence raw clips: {current_original_clip_count(RAW_SILENCE_DIR)}')
    silence_rms = sample_rms_values(RAW_SILENCE_DIR, sample_count=20)
    if silence_rms:
        print('Silence sample RMS: ' + f'mean={np.mean(silence_rms):.6f}, min={np.min(silence_rms):.6f}, max={np.max(silence_rms):.6f}')

    print('\n=== Step 2: Unknown mining ===')
    unknown_saved = mine_unknown(TARGET_CLIPS)
    print(f'Unknown raw clips: {current_original_clip_count(RAW_UNKNOWN_DIR)}')
    unknown_rms = sample_rms_values(RAW_UNKNOWN_DIR, sample_count=20)
    if unknown_rms:
        print('Unknown sample RMS: ' + f'mean={np.mean(unknown_rms):.6f}, min={np.min(unknown_rms):.6f}, max={np.max(unknown_rms):.6f}')

    print('\n=== Step 3: Augmentation ===')
    silence_added = augment_folder(RAW_SILENCE_DIR)
    unknown_added = augment_folder(RAW_UNKNOWN_DIR)
    print(f'Silence total wavs: {current_total_clip_count(RAW_SILENCE_DIR)}')
    print(f'Unknown total wavs: {current_total_clip_count(RAW_UNKNOWN_DIR)}')
    print(f'Augmented added: silence={silence_added}, unknown={unknown_added}')

    print('\n=== Step 4: Split summary ===')
    split = update_split_summary()
    val_split = summarize_split('val')
    test_split = summarize_split('test')
    print(f"Train speakers: {split['num_speakers']}, samples: {split['num_samples']}")
    print(f"Val speakers: {val_split['num_speakers']}, samples: {val_split['num_samples']}")
    print(f"Test speakers: {test_split['num_speakers']}, samples: {test_split['num_samples']}")

    print('\n=== Step 5: Label classes ===')
    labels = label_summary()
    print(f'Label count: {len(labels)}')
    print('Label classes: ' + ', '.join(labels))

    print('\n=== Step 6: New MFCC extraction ===')
    silence_features = extract_new_features(RAW_SILENCE_DIR, PROCESSED_SILENCE_DIR, MFCC_MEAN, MFCC_STD)
    unknown_features = extract_new_features(RAW_UNKNOWN_DIR, PROCESSED_UNKNOWN_DIR, MFCC_MEAN, MFCC_STD)
    print(f'Silence feature files generated: {len(silence_features)}')
    print(f'Unknown feature files generated: {len(unknown_features)}')
    print('Sample silence features: ' + ', '.join(str(path.relative_to(ROOT)) for path in silence_features[:5]))
    print('Sample unknown features: ' + ', '.join(str(path.relative_to(ROOT)) for path in unknown_features[:5]))

    print('\n=== Final checks ===')
    print(f'Silence raw target reached: {current_original_clip_count(RAW_SILENCE_DIR)} / {TARGET_CLIPS}')
    print(f'Unknown raw target reached: {current_original_clip_count(RAW_UNKNOWN_DIR)} / {TARGET_CLIPS}')
    print(f'Silence feature count: {count_feature_files(PROCESSED_SILENCE_DIR)}')
    print(f'Unknown feature count: {count_feature_files(PROCESSED_UNKNOWN_DIR)}')
    print('STOP HERE: no model retraining was run.')


if __name__ == '__main__':
    main()
