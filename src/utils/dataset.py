import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'saved')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DEFAULT_SPLIT_PATH = os.path.join(PROJECT_ROOT, 'configs', 'speaker_split_v1.json')


def load_split_config(split_path: str = DEFAULT_SPLIT_PATH) -> Dict:
    with open(split_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def get_split_speakers(split_name: str, split_config: Dict | None = None) -> List[str]:
    config = split_config or load_split_config()

    if split_name == 'train':
        return config['splits']['train_complete'] + config['splits'].get('train_partial', [])

    if split_name in {'val', 'test'}:
        return list(config['splits'][split_name])

    raise ValueError(f'Unknown split: {split_name}')


def load_dataset_for_speakers(
    speakers: Sequence[str],
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[str] = []
    speaker_ids: List[str] = []

    for speaker in speakers:
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


def load_split_dataset(
    split_name: str,
    split_path: str = DEFAULT_SPLIT_PATH,
    processed_dir: str = PROCESSED_DIR,
    keywords: Sequence[str] = KEYWORDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
    features, labels, _, _ = load_split_dataset(split_name, split_path, processed_dir, keywords)
    if len(features) == 0:
        return np.array([], dtype=np.float32), []

    label_to_index = {word: index for index, word in enumerate(keywords)}
    limited_features = features[:max_samples].astype(np.float32)[..., np.newaxis]
    limited_labels = [label_to_index[label] for label in labels[:max_samples]]
    return limited_features, limited_labels


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)