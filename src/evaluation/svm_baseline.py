"""SVM baseline for NepSpot v6.

Classic non-deep baseline: RBF-kernel SVM on standardized flattened MFCC
features, evaluated on the speaker-independent test split. No standalone SVM
script survived in the repo, so this reproduces the standard baseline
methodology. Output: results/metrics/svm_baseline_v6.txt
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data.dataset import load_split_dataset  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'metrics', 'svm_baseline_v6.txt')
SEED = 42


def flatten(X):
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(len(X), -1)


def main():
    X_tr, y_tr, _, _ = load_split_dataset('train')
    X_te, y_te, _, _ = load_split_dataset('test')
    X_tr, X_te = flatten(X_tr), flatten(X_te)

    classes = sorted(set(y_tr) | set(y_te))
    idx = {c: i for i, c in enumerate(classes)}
    y_tr_i = np.array([idx[w] for w in y_tr], dtype=np.int64)
    y_te_i = np.array([idx[w] for w in y_te], dtype=np.int64)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = SVC(kernel='rbf', random_state=SEED)
    clf.fit(X_tr_s, y_tr_i)
    pred = clf.predict(X_te_s)

    acc = accuracy_score(y_te_i, pred) * 100.0
    macro_f1 = f1_score(y_te_i, pred, average='macro', zero_division=0)
    report = classification_report(y_te_i, pred, target_names=classes,
                                   digits=4, zero_division=0)

    lines = [
        'SVM baseline - NepSpot v6',
        'Features: flattened MFCC + StandardScaler | Classifier: SVC(kernel=rbf)',
        'Train: %d clips | Test (voicer28/29/30): %d clips' % (len(X_tr), len(X_te)),
        '',
        'Test accuracy: %.2f%%' % acc,
        'Macro F1: %.4f' % macro_f1,
        '',
        report,
    ]
    txt = '\n'.join(lines) + '\n'
    print(txt, end='')
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print('Wrote ' + OUT)


if __name__ == '__main__':
    main()
