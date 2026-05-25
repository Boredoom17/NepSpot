"""ARCHIVED: This script orchestrated the v6 3-seed experiment whose outputs
are saved under models/{saved,tflite}/v6/ and results/seeds_experiment_artifacts/.
The per-architecture trainer scripts it references were removed in repo cleanup.
Retained as documentation of the experiment configuration.

Run N=3 seed × 3 model statistical-rigor experiment.

For each (model, seed) pair, launches the existing training script as a
subprocess with the SEED env var set. Each training script reads SEED at the
top of its prologue (replacing the previously hardcoded literal 42) and
propagates it through random.seed / np.random.seed / tf.random.set_seed /
PYTHONHASHSEED / _training_utils._BASE_SEED — so the seed change is
the only differing input across runs.

After each subprocess completes successfully, this wrapper:
  - Parses the report file written by the training script for INT8 accuracy,
    macro F1, INT8 size, and per-class F1.
  - Archives the TFLite + report under
    results/seeds_experiment_artifacts/<model_key>_seed<seed>.{tflite,txt}
    so the next run (which overwrites the same training-script output paths)
    does not clobber prior results.

After all 9 runs, writes:
  - results/metrics/seeds_experiment_report.txt (formatted table + summary)
  - results/metrics/seeds_experiment_raw.json   (machine-readable record)

Run:
    cd /Users/ad/codes/NepSpot
    /Users/ad/codes/.venv310/bin/python3 scripts/run_seeds_experiment.py

Expected runtime ~2.5 hours (BC-ResNet 32min × 3 + Vanilla 4min × 3 +
DS-CNN 7min × 3). Per-run subprocess logs land at
results/seeds_experiment_artifacts/logs/<model_key>_seed<seed>.log.
"""

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = '/Users/ad/codes/.venv310/bin/python3'

KEYWORDS = [
    'aghillo', 'arko', 'baalnu', 'banda',
    'feri', 'hoina', 'huncha', 'maathi',
    'roknu', 'suru', 'tala', 'thik_chha',
]

MODELS = [
    {
        'name': 'BC-ResNet',
        'key': 'bcresnet',
        'script': 'src/models/train_bcresnet_phase1.py',
        'tflite': 'models/tflite/bc_resnet_int8_phase1.tflite',
        'report': 'results/metrics/bc_resnet_phase1_report.txt',
    },
    {
        'name': 'Vanilla CNN',
        'key': 'vanilla',
        'script': 'src/models/train_vanilla_phase1.py',
        'tflite': 'models/tflite/vanilla_cnn_int8_phase1.tflite',
        'report': 'results/metrics/vanilla_cnn_phase1_report.txt',
    },
    {
        'name': 'DS-CNN',
        'key': 'dscnn',
        'script': 'src/models/train_phase1.py',
        'tflite': 'models/tflite/nepspot_int8_phase1.tflite',
        'report': 'results/metrics/classification_report_phase1.txt',
    },
]

SEEDS = [42, 123, 456]

ARTIFACTS_DIR = PROJECT_ROOT / 'results' / 'seeds_experiment_artifacts'
LOGS_DIR = ARTIFACTS_DIR / 'logs'
TEXT_REPORT_PATH = PROJECT_ROOT / 'results' / 'metrics' / 'seeds_experiment_report.txt'
JSON_REPORT_PATH = PROJECT_ROOT / 'results' / 'metrics' / 'seeds_experiment_raw.json'


# --- Report parser ----------------------------------------------------------


_INT8_ACC_RE = re.compile(r'^INT8 test accuracy:\s*([0-9.]+)\s*%', re.MULTILINE)
_INT8_F1_RE = re.compile(r'^INT8 macro F1:\s*([0-9.]+)', re.MULTILINE)
_INT8_SIZE_RE = re.compile(r'^INT8 model size \(KB\):\s*([0-9.]+)', re.MULTILINE)


def parse_report(report_path: Path) -> dict:
    text = report_path.read_text(encoding='utf-8')

    acc_match = _INT8_ACC_RE.search(text)
    f1_match = _INT8_F1_RE.search(text)
    size_match = _INT8_SIZE_RE.search(text)
    if not (acc_match and f1_match and size_match):
        raise ValueError(
            f'Could not parse INT8 metrics from {report_path}. '
            f'acc_match={bool(acc_match)} f1_match={bool(f1_match)} size_match={bool(size_match)}'
        )

    accuracy = float(acc_match.group(1)) / 100.0
    macro_f1 = float(f1_match.group(1))
    size_kb = float(size_match.group(1))

    # Per-class F1: scan lines whose first token is a keyword.
    per_class_f1: dict = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in KEYWORDS:
            try:
                per_class_f1[parts[0]] = float(parts[3])
            except ValueError:
                continue

    if len(per_class_f1) != len(KEYWORDS):
        missing = set(KEYWORDS) - set(per_class_f1)
        raise ValueError(
            f'Per-class F1 incomplete in {report_path}; missing: {sorted(missing)}'
        )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'int8_size_kb': size_kb,
        'per_class_f1': per_class_f1,
    }


# --- Runner -----------------------------------------------------------------


def run_one(model: dict, seed: int) -> dict:
    script_path = PROJECT_ROOT / model['script']
    tflite_path = PROJECT_ROOT / model['tflite']
    report_path = PROJECT_ROOT / model['report']

    log_path = LOGS_DIR / f"{model['key']}_seed{seed}.log"
    archive_tflite = ARTIFACTS_DIR / f"{model['key']}_seed{seed}.tflite"
    archive_report = ARTIFACTS_DIR / f"{model['key']}_seed{seed}_report.txt"

    # Wipe prior-run output files so a silent re-use of stale artifacts
    # cannot masquerade as a successful run.
    for p in (tflite_path, report_path):
        if p.exists():
            p.unlink()

    env = os.environ.copy()
    env['SEED'] = str(seed)
    env['PYTHONHASHSEED'] = str(seed)
    env['PYTHONUNBUFFERED'] = '1'
    # Force CPU-only training (model fit is already CPU-bound on macOS).
    env.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8') as log_fh:
        log_fh.write(f'# {model["name"]} seed={seed}\n')
        log_fh.write(f'# command: {VENV_PYTHON} {script_path}\n')
        log_fh.write(f'# SEED={seed}\n\n')
        log_fh.flush()
        proc = subprocess.run(
            [VENV_PYTHON, '-u', str(script_path)],
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {
            'status': 'failed',
            'returncode': proc.returncode,
            'elapsed_s': elapsed,
            'log': str(log_path.relative_to(PROJECT_ROOT)),
            'error': f'subprocess exited with returncode {proc.returncode}',
        }

    if not report_path.exists():
        return {
            'status': 'failed',
            'elapsed_s': elapsed,
            'log': str(log_path.relative_to(PROJECT_ROOT)),
            'error': f'report not found at {report_path}',
        }
    if not tflite_path.exists():
        return {
            'status': 'failed',
            'elapsed_s': elapsed,
            'log': str(log_path.relative_to(PROJECT_ROOT)),
            'error': f'tflite not found at {tflite_path}',
        }

    metrics = parse_report(report_path)

    shutil.copy2(tflite_path, archive_tflite)
    shutil.copy2(report_path, archive_report)

    return {
        'status': 'ok',
        'elapsed_s': elapsed,
        'log': str(log_path.relative_to(PROJECT_ROOT)),
        'archive_tflite': str(archive_tflite.relative_to(PROJECT_ROOT)),
        'archive_report': str(archive_report.relative_to(PROJECT_ROOT)),
        **metrics,
    }


# --- Report writer ----------------------------------------------------------


def _format_pct(value, precision=2):
    return f'{value * 100:.{precision}f}%' if value is not None else 'n/a'


def _stdev_safe(values):
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_text_report(all_results: dict, path: Path):
    lines = []
    lines.append('Seeds experiment — N=3 per model, INT8 metrics on speaker-independent test split')
    lines.append('(voicer28, voicer29, voicer30 — originals only, no _aug_ files; 361 samples)')
    lines.append('')
    lines.append('Per-run results')
    lines.append('-' * 64)

    header = f'{"Model":<12}| {"Seed":>4} | {"INT8 Acc":>8} | {"Macro F1":>8} | {"Size KB":>7}'
    sep = '-' * 12 + '+' + '-' * 6 + '+' + '-' * 10 + '+' + '-' * 10 + '+' + '-' * 9
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        runs = all_results.get(model['name'], [])
        for run in runs:
            if run.get('status') == 'ok':
                lines.append(
                    f'{model["name"]:<12}| {run["seed"]:>4} | '
                    f'{_format_pct(run["accuracy"]):>8} | '
                    f'{run["macro_f1"]:>8.4f} | '
                    f'{run["int8_size_kb"]:>7.2f}'
                )
            else:
                lines.append(
                    f'{model["name"]:<12}| {run["seed"]:>4} | {"FAILED":>8} | '
                    f'{"-":>8} | {"-":>7}  ({run.get("error", "?")})'
                )

    lines.append('')
    lines.append('Per-model summary (mean ± std, ddof=1)')
    lines.append('-' * 64)
    s_header = f'{"Model":<12}| {"Mean Acc ± Std":>16} | {"Mean F1 ± Std":>17} | {"Size KB":>8}'
    s_sep = '-' * 12 + '+' + '-' * 18 + '+' + '-' * 19 + '+' + '-' * 10
    lines.append(s_header)
    lines.append(s_sep)
    for model in MODELS:
        ok_runs = [r for r in all_results.get(model['name'], []) if r.get('status') == 'ok']
        if not ok_runs:
            lines.append(f'{model["name"]:<12}| {"no successful runs":>16} | {"-":>17} | {"-":>8}')
            continue
        accs = [r['accuracy'] for r in ok_runs]
        f1s = [r['macro_f1'] for r in ok_runs]
        sizes = [r['int8_size_kb'] for r in ok_runs]
        acc_mean = statistics.fmean(accs)
        acc_std = _stdev_safe(accs)
        f1_mean = statistics.fmean(f1s)
        f1_std = _stdev_safe(f1s)
        size_mean = statistics.fmean(sizes)
        lines.append(
            f'{model["name"]:<12}| '
            f'{_format_pct(acc_mean):>6} ± {_format_pct(acc_std):>6} | '
            f' {f1_mean:>5.4f} ± {f1_std:>5.4f}  | '
            f'{size_mean:>8.2f}'
        )

    lines.append('')
    lines.append('Per-class INT8 F1 (mean across seeds)')
    lines.append('-' * 64)
    pc_header = f'{"Class":<12}|' + ''.join(f' {m["name"]:>12} |' for m in MODELS)
    pc_sep = '-' * 12 + '+' + ('-' * 14 + '+') * len(MODELS)
    lines.append(pc_header)
    lines.append(pc_sep)
    for kw in KEYWORDS:
        row = f'{kw:<12}|'
        for model in MODELS:
            ok_runs = [r for r in all_results.get(model['name'], []) if r.get('status') == 'ok']
            if not ok_runs:
                row += f' {"-":>12} |'
                continue
            f1s = [r['per_class_f1'].get(kw) for r in ok_runs if r['per_class_f1'].get(kw) is not None]
            row += f' {statistics.fmean(f1s):>12.4f} |' if f1s else f' {"-":>12} |'
        lines.append(row)

    lines.append('')
    lines.append('Notes')
    lines.append('-' * 64)
    lines.append(f'Seeds: {SEEDS}')
    lines.append('Each run uses identical recipe; only SEED env var changes.')
    lines.append('SEED propagates to: PYTHONHASHSEED, random.seed, np.random.seed,')
    lines.append('  tf.random.set_seed, tf.data shuffle seed, and')
    lines.append('  _training_utils._BASE_SEED (SpecAugment + Mixup stateless seeds).')
    lines.append(f'Per-run artifacts archived under {ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}/.')
    lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_json_report(all_results: dict, path: Path):
    summary = {}
    for model in MODELS:
        ok_runs = [r for r in all_results.get(model['name'], []) if r.get('status') == 'ok']
        if not ok_runs:
            summary[model['name']] = None
            continue
        accs = [r['accuracy'] for r in ok_runs]
        f1s = [r['macro_f1'] for r in ok_runs]
        sizes = [r['int8_size_kb'] for r in ok_runs]
        per_class_means = {}
        for kw in KEYWORDS:
            vals = [r['per_class_f1'].get(kw) for r in ok_runs if r['per_class_f1'].get(kw) is not None]
            per_class_means[kw] = statistics.fmean(vals) if vals else None
        summary[model['name']] = {
            'n_runs': len(ok_runs),
            'accuracy_mean': statistics.fmean(accs),
            'accuracy_std': _stdev_safe(accs),
            'macro_f1_mean': statistics.fmean(f1s),
            'macro_f1_std': _stdev_safe(f1s),
            'int8_size_kb_mean': statistics.fmean(sizes),
            'per_class_f1_mean': per_class_means,
        }

    payload = {
        'seeds': SEEDS,
        'models': [m['name'] for m in MODELS],
        'keywords': KEYWORDS,
        'runs': all_results,
        'summary': summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


# --- Main -------------------------------------------------------------------


def main():
    if not Path(VENV_PYTHON).exists():
        sys.exit(f'venv python not found: {VENV_PYTHON}')

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict = {m['name']: [] for m in MODELS}

    total = len(MODELS) * len(SEEDS)
    counter = 0
    overall_t0 = time.time()

    for model in MODELS:
        for seed in SEEDS:
            counter += 1
            print(f'[{counter}/{total}] {model["name"]} seed={seed} ... ', end='', flush=True)
            result = run_one(model, seed)
            result['seed'] = seed
            all_results[model['name']].append(result)

            if result['status'] == 'ok':
                print(
                    f'OK in {result["elapsed_s"] / 60:.1f}min  '
                    f'acc={result["accuracy"] * 100:.2f}%  '
                    f'macroF1={result["macro_f1"]:.4f}  '
                    f'size={result["int8_size_kb"]:.1f}KB'
                )
            else:
                print(f'FAILED: {result.get("error")}  (log: {result["log"]})')

            # Write incremental reports after every run so a mid-experiment
            # crash still leaves usable artifacts on disk.
            write_text_report(all_results, TEXT_REPORT_PATH)
            write_json_report(all_results, JSON_REPORT_PATH)

    total_min = (time.time() - overall_t0) / 60.0
    print(f'\nAll runs done in {total_min:.1f} min')

    # Final summary print (only the per-model means).
    print('\nFinal per-model INT8 summary (mean ± std):')
    for model in MODELS:
        ok_runs = [r for r in all_results.get(model['name'], []) if r.get('status') == 'ok']
        if not ok_runs:
            print(f'  {model["name"]:<12}  no successful runs')
            continue
        accs = [r['accuracy'] for r in ok_runs]
        f1s = [r['macro_f1'] for r in ok_runs]
        acc_mean = statistics.fmean(accs)
        acc_std = _stdev_safe(accs)
        f1_mean = statistics.fmean(f1s)
        f1_std = _stdev_safe(f1s)
        print(
            f'  {model["name"]:<12}  '
            f'acc {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%   '
            f'F1 {f1_mean:.4f} ± {f1_std:.4f}   (n={len(ok_runs)})'
        )

    print(f'\nText report: {TEXT_REPORT_PATH.relative_to(PROJECT_ROOT)}')
    print(f'Raw JSON:    {JSON_REPORT_PATH.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
