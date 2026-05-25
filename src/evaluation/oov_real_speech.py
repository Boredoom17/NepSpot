"""OOV (out-of-vocabulary) rejection test for NepSpot v6.

Runs BC-ResNet seed 42 (post-QAT float32 SavedModel at the default path) on
the 100 OOV Nepali speech clips under data/raw/oov_real_speech_test/. None of
the clips contain any of the 12 target keywords, so a well-behaved KWS model
should produce low max-softmax probability and be rejected by any reasonable
confidence threshold.

Output: results/metrics/oov_real_speech_v6.txt (matches v5 format).
"""

import os
from pathlib import Path

import librosa
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ROOT = Path(__file__).resolve().parents[2]
OOV_DIR = ROOT / 'data' / 'raw' / 'oov_real_speech_test'
MEAN_PATH = ROOT / 'models' / 'saved' / 'mfcc_mean.npy'
STD_PATH = ROOT / 'models' / 'saved' / 'mfcc_std.npy'
MODEL_PATH = ROOT / 'models' / 'saved' / 'v6' / 'bcresnet_seed42_saved_model'
OUT = ROOT / 'results' / 'metrics' / 'oov_real_speech_v6.txt'

SR = 16000
DURATION = 1.0
N_MFCC = 40
HOP = 512
N_FFT = 1024
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95


def load_and_pad(path):
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    target = int(SR * DURATION)
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    elif len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode='constant')
    return audio


def extract_mfcc(audio):
    return librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                hop_length=HOP, n_fft=N_FFT)


def main():
    import tensorflow as tf

    mean = float(np.load(MEAN_PATH))
    std = float(np.load(STD_PATH))

    wavs = sorted(OOV_DIR.glob('*.wav'))
    if not wavs:
        raise SystemExit('No OOV wavs in %s' % OOV_DIR)

    X = []
    for w in wavs:
        audio = load_and_pad(w)
        m = extract_mfcc(audio)
        m_norm = (m - mean) / (std + 1e-8)
        X.append(m_norm.astype(np.float32))
    X = np.stack(X, axis=0)[..., np.newaxis]   # (N, 40, 32, 1)

    sm = tf.saved_model.load(str(MODEL_PATH))
    sig = sm.signatures['serving_default']
    in_key = list(sig.structured_input_signature[1].keys())[0]
    out_key = list(sig.structured_outputs.keys())[0]
    out_batches = []
    for i in range(0, len(X), 32):
        b = tf.constant(X[i:i + 32])
        out_batches.append(sig(**{in_key: b})[out_key].numpy())
    probs = np.concatenate(out_batches, axis=0)

    row_sum = probs.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-3):
        z = probs - probs.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    max_conf = probs.max(axis=1)
    n = len(max_conf)

    lines = []
    lines.append('OOV Real Speech Test (NepSpot v6)')
    lines.append('=' * 34)
    lines.append('')
    lines.append('Dataset: %d real Nepali speech utterances with NO target keywords' % n)
    lines.append('Source: data/raw/oov_real_speech_test/')
    lines.append('Model: BC-ResNet seed 42 (post-QAT float32 SavedModel)')
    lines.append('Duration per clip: %.1f second @ %d kHz mono' % (DURATION, SR // 1000))
    lines.append('')
    lines.append('Max softmax confidence across all %d OOV clips: %.3f' % (n, float(max_conf.max())))
    lines.append('Mean max-confidence: %.3f   Median: %.3f'
                 % (float(max_conf.mean()), float(np.median(max_conf))))
    lines.append('')
    lines.append('Threshold Sweep Results:')
    lines.append('Threshold | FAR (%) | FRR (%) | Accepted | Rejected')
    lines.append('-' * 54)
    for thr in THRESHOLDS:
        accepted = int((max_conf >= thr).sum())
        rejected = n - accepted
        far = accepted / n * 100.0
        frr = rejected / n * 100.0
        lines.append('%.2f      | %6.2f  | %6.2f  | %8d | %8d'
                     % (thr, far, frr, accepted, rejected))

    lines.append('')
    lines.append('Analysis:')
    lines.append('  - Max confidence over all OOV clips: %.3f' % float(max_conf.max()))
    lines.append('  - FAR at threshold 0.30: %.2f%%' % ((max_conf >= 0.30).sum() / n * 100.0))
    lines.append('  - FAR at threshold 0.95: %.2f%%' % ((max_conf >= 0.95).sum() / n * 100.0))

    txt = '\n'.join(lines) + '\n'
    print(txt, end='')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print('Wrote ' + str(OUT))


if __name__ == '__main__':
    main()
