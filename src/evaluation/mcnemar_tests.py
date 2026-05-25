"""McNemar pairwise significance tests for NepSpot v6.

Compares the three seed-42 float32 models pairwise on the speaker-independent
test split. Reuses the per-sample predictions cached by bootstrap_ci.py
(results/metrics/bootstrap_predictions.npz), so no re-inference is needed.

Run after bootstrap_ci.py. Output: results/metrics/mcnemar_tests_v6.txt
"""

import os

import numpy as np
from scipy.stats import binomtest, chi2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDS = os.path.join(ROOT, 'results', 'metrics', 'bootstrap_predictions.npz')
OUT = os.path.join(ROOT, 'results', 'metrics', 'mcnemar_tests_v6.txt')

if not os.path.exists(PREDS):
    raise SystemExit('Missing %s - run scripts/bootstrap_ci.py first.' % PREDS)

data = np.load(PREDS, allow_pickle=False)
y = data['y_true']
models = {
    'BC-ResNet': data['BC-ResNet'],
    'DS-CNN': data['DS-CNN'],
    'Vanilla CNN': data['Vanilla CNN'],
}
correct = {k: (v == y) for k, v in models.items()}

pairs = [('BC-ResNet', 'DS-CNN'), ('BC-ResNet', 'Vanilla CNN'), ('DS-CNN', 'Vanilla CNN')]

lines = []
lines.append('McNemar pairwise tests - NepSpot v6')
lines.append('Test set: voicer28/29/30, n=%d clips, 12 keywords' % len(y))
lines.append('Models: seed-42 float32 checkpoints (predictions cached by bootstrap_ci.py)')
lines.append('')

for a, b in pairs:
    ca, cb = correct[a], correct[b]
    n01 = int(np.sum((~ca) & cb))   # a wrong, b correct
    n10 = int(np.sum(ca & (~cb)))   # a correct, b wrong
    n_disc = n01 + n10
    if n_disc > 0:
        p_exact = float(binomtest(n01, n_disc, 0.5, alternative='two-sided').pvalue)
        stat = (abs(n10 - n01) - 1) ** 2 / n_disc
        p_chi = float(chi2.sf(stat, 1))
    else:
        p_exact, stat, p_chi = 1.0, 0.0, 1.0
    lines.append('%s vs %s' % (a, b))
    lines.append('  %-12s correct: %3d/%d (%.2f%%)' % (a, int(ca.sum()), len(y), 100 * ca.mean()))
    lines.append('  %-12s correct: %3d/%d (%.2f%%)' % (b, int(cb.sum()), len(y), 100 * cb.mean()))
    lines.append('  discordant: %s-only-wrong=%d, %s-only-wrong=%d, total=%d'
                 % (a, n01, b, n10, n_disc))
    lines.append('  McNemar exact binomial p = %.3f' % p_exact)
    lines.append('  McNemar chi-square (continuity-corrected) stat=%.3f, p = %.3f' % (stat, p_chi))
    lines.append('')

txt = '\n'.join(lines) + '\n'
print(txt, end='')
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write(txt)
print('Wrote ' + OUT)
