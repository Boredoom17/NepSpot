"""Bootstrap 95% confidence intervals for accuracy and macro F1.

Runs float32 inference for Vanilla CNN, DS-CNN, and BC-ResNet on the
speaker-independent test split (voicer28/29/30, 361 clips, 12 keywords),
caches per-sample predictions to results/metrics/bootstrap_predictions.npz,
then computes percentile bootstrap CIs (1000 resamples, seed=42).
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data.dataset import load_split_dataset  # noqa: E402

MODELS = [
    ('Vanilla CNN', 'models/saved/v6/vanilla_seed42_best.keras',  'keras'),
    ('DS-CNN',      'models/saved/v6/dscnn_seed42_best.keras',    'keras'),
    ('BC-ResNet',   'models/saved/v6/bcresnet_seed42_saved_model', 'savedmodel'),
]

PREDS_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'bootstrap_predictions.npz')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'bootstrap_ci.txt')

N_BOOT = 1000
SEED = 42


def build_predictions():
    import tensorflow as tf

    X, y_str, _, _ = load_split_dataset('test')
    X = X[..., np.newaxis].astype(np.float32)

    sorted_classes = sorted(set(y_str))
    label_to_idx = {w: i for i, w in enumerate(sorted_classes)}
    y_true = np.array([label_to_idx[w] for w in y_str], dtype=np.int64)

    preds = {'y_true': y_true, 'classes': np.array(sorted_classes)}
    for name, rel_path, kind in MODELS:
        path = os.path.join(PROJECT_ROOT, rel_path)
        if kind == 'keras':
            model = tf.keras.models.load_model(path, compile=False)
            logits = model.predict(X, batch_size=32, verbose=0)
        elif kind == 'savedmodel':
            sm = tf.saved_model.load(path)
            sig = sm.signatures['serving_default']
            in_key = list(sig.structured_input_signature[1].keys())[0]
            out_key = list(sig.structured_outputs.keys())[0]
            batches = []
            for i in range(0, len(X), 32):
                batch = tf.constant(X[i:i + 32])
                batches.append(sig(**{in_key: batch})[out_key].numpy())
            logits = np.concatenate(batches, axis=0)
        else:
            raise ValueError(f'unknown kind: {kind}')
        preds[name] = np.argmax(logits, axis=1).astype(np.int64)
    np.savez(PREDS_PATH, **preds)
    return preds


def load_predictions():
    if not os.path.exists(PREDS_PATH):
        return build_predictions()
    data = np.load(PREDS_PATH, allow_pickle=False)
    keys = set(data.files)
    if not all(name in keys for name, _, _ in MODELS) or 'y_true' not in keys:
        return build_predictions()
    return {k: data[k] for k in data.files}


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = np.empty(n_boot, dtype=np.float64)
    f1s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        accs[b] = accuracy_score(yt, yp)
        f1s[b] = f1_score(yt, yp, average='macro', zero_division=0)
    return accs, f1s


def main():
    preds = load_predictions()
    y_true = preds['y_true']

    lines = []
    lines.append('Bootstrap 95% confidence intervals')
    lines.append(f'Test set: voicer28/29/30, n={len(y_true)} clips, 12 keywords')
    lines.append(f'Resamples: {N_BOOT}, seed: {SEED}')
    lines.append(f'Models evaluated as float32 .keras checkpoints')
    lines.append('')
    lines.append('Accuracy')
    lines.append('--------')
    for name, _, _ in MODELS:
        y_pred = preds[name]
        point = accuracy_score(y_true, y_pred) * 100.0
        accs, _ = bootstrap_ci(y_true, y_pred)
        lo, hi = np.percentile(accs, [2.5, 97.5]) * 100.0
        lines.append(f'{name:<12}: {point:5.2f}% [{lo:5.2f}%, {hi:5.2f}%]')

    lines.append('')
    lines.append('Macro F1')
    lines.append('--------')
    for name, _, _ in MODELS:
        y_pred = preds[name]
        point = f1_score(y_true, y_pred, average='macro', zero_division=0)
        _, f1s = bootstrap_ci(y_true, y_pred)
        lo, hi = np.percentile(f1s, [2.5, 97.5])
        lines.append(f'{name:<12}: {point:.4f} [{lo:.4f}, {hi:.4f}]')

    out = '\n'.join(lines) + '\n'
    print(out, end='')
    with open(RESULTS_PATH, 'w', encoding='utf-8') as fh:
        fh.write(out)


if __name__ == '__main__':
    main()
