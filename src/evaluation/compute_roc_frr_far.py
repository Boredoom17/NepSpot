"""FRR / FAR / ROC analysis — BC-ResNet INT8 (new architecture, seed 42).

Treats each keyword as a one-vs-rest binary detector over the speaker-
independent test split (voicer28-30, originals only). Uses the INT8 per-class
score (no softmax / no threshold) as the detector score.

For each keyword:
  - Sweeps thresholds via sklearn.roc_curve → FAR (false accept rate) and
    TPR (true positive rate). FRR = 1 - TPR.
  - AUC = area under the ROC curve.
  - EER = the threshold where FAR ≈ FRR (interpolated).
  - FRR @ FAR target (0.1%, 1%, 5%) = the missed-detect rate when the
    threshold is set to the smallest value that keeps FAR ≤ target.

Macro averages are unweighted means across the 12 keywords; macro-ROC is
also produced by averaging TPR over a fixed FAR grid (for the figure only).

Run from NepSpot root:
    /Users/ad/codes/.venv310/bin/python3 compute_roc_frr_far.py

Outputs (overwrites previous):
    results/metrics/frr_far_roc_new.txt
    results/figures/roc_curves_new.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'models' / 'tflite' / 'v6' / 'bcresnet_seed42_int8.tflite'
DATA_DIR = ROOT / 'data' / 'processed'
SPLIT_CONFIG = ROOT / 'configs' / 'speaker_split.json'
LABELS_PATH = ROOT / 'models' / 'saved' / 'label_classes.npy'

OUTPUT_TXT = ROOT / 'results' / 'metrics' / 'frr_far_roc_new.txt'
OUTPUT_FIG = ROOT / 'results' / 'figures' / 'roc_curves_new.png'

FAR_TARGETS = [0.001, 0.01, 0.05]  # 0.1%, 1%, 5%
PRIMARY_FAR_TARGET = 0.01           # the headline number the user asked for


def load_test_data(label_classes):
    with open(SPLIT_CONFIG, 'r') as fh:
        splits = json.load(fh)
    test_speakers = [s for s in splits['splits']['test'] if not s.startswith('_')]

    X, y = [], []
    for speaker in test_speakers:
        speaker_dir = DATA_DIR / speaker
        if not speaker_dir.is_dir():
            continue
        for label_idx, label in enumerate(label_classes):
            label_dir = speaker_dir / label
            if not label_dir.is_dir():
                continue
            for npy_file in sorted(label_dir.glob('*.npy')):
                # data/processed/ files are already mean/std normalised
                # by src/features/extract_mfcc.py — feed them through unchanged.
                if '_aug_' in npy_file.name:
                    continue
                X.append(np.load(npy_file))
                y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), test_speakers


def run_inference(interpreter, X):
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    if in_scale == 0:
        raise RuntimeError('INT8 input scale is 0 — model is not properly quantized')

    out_scale, out_zp = out_det['quantization']

    scores_int8 = np.empty((len(X), out_det['shape'][-1]), dtype=np.int8)
    for i, mfcc in enumerate(X):
        q = np.round(mfcc / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(in_det['index'], q[np.newaxis, :, :, np.newaxis])
        interpreter.invoke()
        scores_int8[i] = interpreter.get_tensor(out_det['index'])[0]

    # Dequantize for ROC (monotonic transform — does not change AUC/EER/FAR-FRR
    # ranking — but makes thresholds human-readable as probabilities).
    if out_scale > 0:
        scores = (scores_int8.astype(np.float32) - out_zp) * out_scale
    else:
        scores = scores_int8.astype(np.float32)
    return scores, in_scale, in_zp, out_scale, out_zp


def eer_from_curve(fpr, tpr, thresholds):
    fnr = 1.0 - tpr
    diff = fnr - fpr
    # Find the crossing point of FNR and FPR.
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        i = int(np.argmin(np.abs(diff)))
        return (fnr[i] + fpr[i]) / 2.0, float(thresholds[i])
    i = int(sign_change[0])
    # Linear interp between (fpr[i], fnr[i]) and (fpr[i+1], fnr[i+1]).
    d1, d2 = diff[i], diff[i + 1]
    if d1 == d2:
        alpha = 0.0
    else:
        alpha = d1 / (d1 - d2)
    eer = float(fnr[i] + alpha * (fnr[i + 1] - fnr[i]))
    thr = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    return eer, thr


def frr_at_fixed_far(fpr, tpr, target_far):
    """Return (FRR, threshold_index) at the operating point with FAR ≤ target_far.

    sklearn returns ROC arrays with thresholds in DECREASING order, so FAR is
    non-decreasing along the array. The strictest operating point that still
    satisfies FAR ≤ target is the LAST index where fpr ≤ target.
    """
    ok = np.where(fpr <= target_far)[0]
    if len(ok) == 0:
        return 1.0, 0
    i = int(ok[-1])
    return float(1.0 - tpr[i]), i


def main():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    label_classes = list(np.load(LABELS_PATH))
    X_test, y_test, test_speakers = load_test_data(label_classes)
    scores, in_scale, in_zp, out_scale, out_zp = run_inference(interpreter, X_test)

    # --- Per-keyword ROC stats ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    per_keyword = []
    for k_idx, kw in enumerate(label_classes):
        y_bin = (y_test == k_idx).astype(int)
        s = scores[:, k_idx]
        fpr, tpr, thr = roc_curve(y_bin, s)
        roc_auc = float(auc(fpr, tpr))
        eer, eer_thr = eer_from_curve(fpr, tpr, thr)

        frr_at_target = {}
        for target in FAR_TARGETS:
            frr, idx_at = frr_at_fixed_far(fpr, tpr, target)
            frr_at_target[target] = {
                'frr': frr,
                'threshold': float(thr[idx_at]),
                'achieved_far': float(fpr[idx_at]),
            }

        per_keyword.append({
            'name': kw,
            'support_positive': int(y_bin.sum()),
            'support_negative': int(len(y_bin) - y_bin.sum()),
            'auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_thr,
            'frr_at_far': frr_at_target,
            'fpr': fpr,
            'tpr': tpr,
        })

        ax.plot(fpr, tpr, lw=1.4, alpha=0.75,
                label=f'{kw} (AUC={roc_auc:.3f}, EER={eer * 100:.1f}%)')

    # --- Macro-ROC for figure -----------------------------------------------
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(mean_fpr)
    for pk in per_keyword:
        mean_tpr += np.interp(mean_fpr, pk['fpr'], pk['tpr'])
    mean_tpr /= len(per_keyword)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    macro_roc_auc = float(auc(mean_fpr, mean_tpr))

    ax.plot(mean_fpr, mean_tpr, 'k-', lw=2.5,
            label=f'Macro-average (AUC={macro_roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.0, alpha=0.4, label='Random')

    # Mark the 1% FAR vertical line.
    ax.axvline(PRIMARY_FAR_TARGET, color='gray', lw=0.8, ls=':', alpha=0.7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('False Acceptance Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (= 1 − FRR)', fontsize=12)
    ax.set_title('BC-ResNet INT8 — per-keyword ROC (12-class Nepali KWS)\n'
                 'Speaker-independent test split (voicer28-30, n=361)', fontsize=12)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Aggregates over keywords -------------------------------------------
    aucs = np.array([pk['auc'] for pk in per_keyword])
    eers = np.array([pk['eer'] for pk in per_keyword])
    frrs_primary = np.array([pk['frr_at_far'][PRIMARY_FAR_TARGET]['frr'] for pk in per_keyword])

    macro = {
        'auc_mean': float(aucs.mean()),
        'auc_std': float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        'eer_mean': float(eers.mean()),
        'eer_std': float(eers.std(ddof=1)) if len(eers) > 1 else 0.0,
        f'frr_at_far_{PRIMARY_FAR_TARGET}_mean': float(frrs_primary.mean()),
        f'frr_at_far_{PRIMARY_FAR_TARGET}_std': float(frrs_primary.std(ddof=1))
        if len(frrs_primary) > 1 else 0.0,
        'macro_roc_auc': macro_roc_auc,
    }

    # --- Text report --------------------------------------------------------
    lines = []
    lines.append('=' * 78)
    lines.append('FRR / FAR / ROC — BC-ResNet INT8 (new architecture, seed 42)')
    lines.append('=' * 78)
    lines.append(f'Model:           {MODEL_PATH.relative_to(ROOT)}')
    lines.append(f'Test split:      voicer28-30 originals (no _aug_ files)')
    lines.append(f'Test samples:    {len(X_test)}')
    lines.append(f'Test speakers:   {", ".join(test_speakers)}')
    lines.append(f'INT8 input qparams:  scale={in_scale:.6f}  zero_point={in_zp}')
    lines.append(f'INT8 output qparams: scale={out_scale:.6f}  zero_point={out_zp}')
    lines.append('')

    lines.append('Per-keyword (FRR/FAR/EER as fractions; AUC dimensionless)')
    lines.append('-' * 78)
    header = (
        f'{"Keyword":<11} {"AUC":>7} {"EER":>7} {"EER thr":>9}  '
        f'{"FRR@FAR=0.1%":>13}  {"FRR@FAR=1%":>11}  {"FRR@FAR=5%":>11}'
    )
    lines.append(header)
    lines.append('-' * len(header))
    for pk in per_keyword:
        f01 = pk['frr_at_far'][0.001]['frr']
        f1 = pk['frr_at_far'][0.01]['frr']
        f5 = pk['frr_at_far'][0.05]['frr']
        lines.append(
            f'{pk["name"]:<11} {pk["auc"]:>7.4f} {pk["eer"] * 100:>6.2f}% '
            f'{pk["eer_threshold"]:>9.3f}  {f01 * 100:>12.2f}%  {f1 * 100:>10.2f}%  '
            f'{f5 * 100:>10.2f}%'
        )

    lines.append('')
    lines.append('Macro averages (unweighted across 12 keywords)')
    lines.append('-' * 78)
    lines.append(f'  AUC (per-keyword mean ± std):        '
                 f'{macro["auc_mean"]:.4f} ± {macro["auc_std"]:.4f}')
    lines.append(f'  AUC of macro-ROC (interp on FAR grid): {macro_roc_auc:.4f}')
    lines.append(f'  EER  (per-keyword mean ± std):       '
                 f'{macro["eer_mean"] * 100:.2f}% ± {macro["eer_std"] * 100:.2f}%')
    lines.append(
        f'  FRR @ FAR=1% (per-keyword mean ± std): '
        f'{macro[f"frr_at_far_{PRIMARY_FAR_TARGET}_mean"] * 100:.2f}% '
        f'± {macro[f"frr_at_far_{PRIMARY_FAR_TARGET}_std"] * 100:.2f}%'
    )
    lines.append('')
    lines.append('Operating-point summary at 1% FAR (per-keyword)')
    lines.append('-' * 78)
    lines.append(f'{"Keyword":<11} {"FRR":>8} {"Threshold (logit)":>20} {"Achieved FAR":>14}')
    lines.append('-' * 78)
    for pk in per_keyword:
        opt = pk['frr_at_far'][PRIMARY_FAR_TARGET]
        lines.append(
            f'{pk["name"]:<11} {opt["frr"] * 100:>7.2f}% '
            f'{opt["threshold"]:>20.4f} {opt["achieved_far"] * 100:>13.3f}%'
        )

    lines.append('')
    lines.append(f'Figure: {OUTPUT_FIG.relative_to(ROOT)}')
    lines.append('')

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f'\nWrote:\n  {OUTPUT_TXT}\n  {OUTPUT_FIG}')


if __name__ == '__main__':
    main()
