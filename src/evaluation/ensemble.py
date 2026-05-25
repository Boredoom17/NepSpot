"""BC-ResNet 3-seed majority-vote ensemble for NepSpot v6.

Hard majority vote over the seed-42/123/456 BC-ResNet saved models on the
speaker-independent test split. 3-way ties are broken by mean-probability
argmax. Output: results/metrics/ensemble_v6.txt
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data.dataset import load_split_dataset  # noqa: E402

SEEDS = [42, 123, 456]
SAVED = [os.path.join(ROOT, 'models', 'saved', 'v6', 'bcresnet_seed%d_saved_model' % s)
         for s in SEEDS]
OUT = os.path.join(ROOT, 'results', 'metrics', 'ensemble_v6.txt')


def predict_savedmodel(path, X):
    import tensorflow as tf
    sm = tf.saved_model.load(path)
    sig = sm.signatures['serving_default']
    in_key = list(sig.structured_input_signature[1].keys())[0]
    out_key = list(sig.structured_outputs.keys())[0]
    out = []
    for i in range(0, len(X), 32):
        batch = tf.constant(X[i:i + 32])
        out.append(sig(**{in_key: batch})[out_key].numpy())
    return np.concatenate(out, axis=0)


def main():
    X, y_str, _, _ = load_split_dataset('test')
    X = X[..., np.newaxis].astype(np.float32)
    classes = sorted(set(y_str))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([label_to_idx[w] for w in y_str], dtype=np.int64)

    probs = [predict_savedmodel(p, X) for p in SAVED]
    preds = [np.argmax(p, axis=1) for p in probs]
    mean_prob = np.mean(probs, axis=0)

    stack = np.stack(preds, axis=1)
    ens = np.empty(len(y_true), dtype=np.int64)
    ties = 0
    for i in range(len(y_true)):
        vals, counts = np.unique(stack[i], return_counts=True)
        if counts.max() >= 2:
            ens[i] = vals[int(np.argmax(counts))]
        else:
            ens[i] = int(np.argmax(mean_prob[i]))
            ties += 1

    acc = accuracy_score(y_true, ens) * 100.0
    macro_f1 = f1_score(y_true, ens, average='macro', zero_division=0)
    report = classification_report(y_true, ens, target_names=classes,
                                   digits=4, zero_division=0)

    lines = []
    lines.append('BC-ResNet 3-seed majority-vote ensemble - NepSpot v6')
    lines.append('Seeds: 42, 123, 456 | Test set: voicer28/29/30, n=%d clips' % len(y_true))
    lines.append('Tie-break (3-way disagreement): mean-probability argmax (%d samples)' % ties)
    lines.append('')
    for seed, pred in zip(SEEDS, preds):
        lines.append('  BC-ResNet seed %3d accuracy: %.2f%%'
                     % (seed, accuracy_score(y_true, pred) * 100.0))
    lines.append('')
    lines.append('Ensemble accuracy: %.2f%%' % acc)
    lines.append('Ensemble macro F1: %.4f' % macro_f1)
    lines.append('')
    lines.append('Per-class report:')
    lines.append(report)

    txt = '\n'.join(lines) + '\n'
    print(txt, end='')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print('Wrote ' + OUT)


if __name__ == '__main__':
    main()
