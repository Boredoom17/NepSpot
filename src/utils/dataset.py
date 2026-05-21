import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

KEYWORDS = [
    'aghillo', 'arko', 'baalnu', 'banda',
    'feri', 'hoina', 'huncha', 'maathi',
    'roknu', 'suru', 'tala', 'thik_chha',
]

EXCLUDED_SPEAKERS = {'_silence', '_unknown', '_silence_test', '_unknown_test'}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DEFAULT_SPLIT_PATH = os.path.join(PROJECT_ROOT, 'configs', 'speaker_split_v1.json')


def load_split_config(split_path: str = DEFAULT_SPLIT_PATH) -> Dict:
    """Load speaker split JSON config from `split_path`.

    Returns the parsed JSON as a dict.
    """
    with open(split_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def get_split_speakers(split_name: str, split_config: Dict | None = None) -> List[str]:
    """Return a list of speaker folder names for a named split.

    `split_name` should be one of 'train', 'val', or 'test'.
    """
    config = split_config or load_split_config()

    if split_name == 'train':
        speakers = config['splits']['train_complete'] + config['splits'].get('train_partial', [])
    elif split_name in {'val', 'test'}:
        speakers = list(config['splits'][split_name])
    else:
        raise ValueError(f'Unknown split: {split_name}')

    return [s for s in speakers if s not in EXCLUDED_SPEAKERS]


def load_dataset_for_speakers(
    speakers: Sequence[str],
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, labels, and speaker IDs for the given speakers.

    How: searches `processed_dir/<speaker>/<keyword>/*.npy` and stacks arrays.
    Returns (features, labels, speaker_ids) as numpy arrays.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []
    speaker_ids: List[str] = []

    for speaker in speakers:
        if speaker in EXCLUDED_SPEAKERS:
            continue
        speaker_dir = os.path.join(processed_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        for word in keywords:
            word_dir = os.path.join(speaker_dir, word)
            if not os.path.isdir(word_dir):
                continue

            for filename in sorted(os.listdir(word_dir)):
                if not filename.endswith('.npy'):
                    continue

                path = os.path.join(word_dir, filename)
                try:
                    mfcc = np.load(path)
                except Exception as exc:
                    print('Failed to load ' + path + ': ' + str(exc))
                    continue

                features.append(mfcc)
                labels.append(word)
                speaker_ids.append(speaker)

    return np.array(features), np.array(labels), np.array(speaker_ids)


def load_dataset_filtered(
    speakers: Sequence[str],
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
    exclude_augmented: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Like load_dataset_for_speakers but with an explicit aug filter.

    When ``exclude_augmented=True`` (the default), any .npy whose filename
    contains the substring ``_aug_`` is skipped. Use this for VAL and TEST
    splits so that augmented copies of speakers cannot leak into evaluation.
    Use ``exclude_augmented=False`` for TRAIN splits where the augmented
    files are wanted.

    Returns the same (features, labels, speaker_ids) tuple as
    load_dataset_for_speakers.
    """
    features: List[np.ndarray] = []
    labels: List[str] = []
    speaker_ids: List[str] = []

    for speaker in speakers:
        if speaker in EXCLUDED_SPEAKERS:
            continue
        speaker_dir = os.path.join(processed_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        for word in keywords:
            word_dir = os.path.join(speaker_dir, word)
            if not os.path.isdir(word_dir):
                continue

            for filename in sorted(os.listdir(word_dir)):
                if not filename.endswith('.npy'):
                    continue
                if exclude_augmented and '_aug_' in filename:
                    continue

                path = os.path.join(word_dir, filename)
                try:
                    mfcc = np.load(path)
                except Exception as exc:
                    print('Failed to load ' + path + ': ' + str(exc))
                    continue

                features.append(mfcc)
                labels.append(word)
                speaker_ids.append(speaker)

    return np.array(features), np.array(labels), np.array(speaker_ids)


def load_split_dataset(
    split_name: str,
    split_path: str = DEFAULT_SPLIT_PATH,
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load dataset arrays for a named split and return the speaker list.

    Returns (features, labels, speaker_ids, speakers).
    """
    config = load_split_config(split_path)
    speakers = get_split_speakers(split_name, config)
    features, labels, speaker_ids = load_dataset_for_speakers(speakers, processed_dir, keywords)
    return features, labels, speaker_ids, speakers


def summarize_split(
    split_name: str,
    split_path: str = DEFAULT_SPLIT_PATH,
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Dict:
    """Return a small summary dict for `split_name` with counts per label and speaker."""
    _, labels, speaker_ids, speakers = load_split_dataset(split_name, split_path, processed_dir, keywords)

    label_counts = {word: int(np.sum(labels == word)) for word in keywords}
    speaker_counts = {
        speaker: int(np.sum(speaker_ids == speaker))
        for speaker in speakers
    }

    return {
        'split': split_name,
        'num_speakers': len(speakers),
        'num_samples': int(len(labels)),
        'label_counts': label_counts,
        'speaker_counts': speaker_counts,
    }


def representative_sample_generator(
    split_name: str = 'train',
    max_samples: int = 200,
    split_path: str = DEFAULT_SPLIT_PATH,
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Tuple[np.ndarray, List[int]]:
    """Return up to `max_samples` features and integer labels for representive sampling.

    Used for TFLite INT8 representative datasets and quick model inspections.
    """
    features, labels, _, _ = load_split_dataset(split_name, split_path, processed_dir, keywords)
    if len(features) == 0:
        return np.array([], dtype=np.float32), []

    label_to_index = {word: index for index, word in enumerate(keywords)}
    limited_features = features[:max_samples].astype(np.float32)[..., np.newaxis]
    limited_labels = [label_to_index[label] for label in labels[:max_samples]]
    return limited_features, limited_labels


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)