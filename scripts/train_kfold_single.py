"""Train ONE model on ONE k-fold split. Subprocess-friendly single-shot trainer.

Usage:
    python scripts/train_kfold_single.py <model> <fold_number>

  <model>        : vanilla | dscnn | bcresnet
  <fold_number>  : 1 | 2 | 3 | 4 | 5

Loads configs/kfold/fold_{N}.json (built by scripts/build_kfold_splits.py)
to get speaker assignments, then runs the same Phase 1 recipe as the
canonical training scripts (SpecAug + Mixup train pipeline, Adam 1e-3 with
label-smoothing 0.1, 50 max epochs, ModelCheckpoint / EarlyStopping(patience=10) /
ReduceLROnPlateau(patience=5), PTQ INT8 via SavedModel — BC-ResNet adds 15
epochs of tfmot QAT fine-tuning and converts the QAT model via
from_keras_model with the experimental flags that handle the broadcast).

Aug policy:
  - TRAIN loads via load_dataset_for_speakers (no filter — _aug_*.npy included)
  - VAL   loads via load_dataset_filtered(..., exclude_augmented=True)
  - TEST  loads via load_dataset_filtered(..., exclude_augmented=True)

Outputs (per (model, fold)):
  - models/kfold/{model}_fold{N}_int8.tflite
  - results/kfold/{model}_fold{N}_report.txt

Console output is one line at the end:
    [<Model> fold N/5] INT8 acc: XX.XX%, F1: 0.XXXX, time: XX min

Run via venv:
    /Users/ad/codes/.venv310/bin/python3 scripts/train_kfold_single.py vanilla 1
"""

import argparse
import gc
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path


# ---- CLI / determinism prologue (must run BEFORE TF import) ----------------

MODEL_CHOICES = ('vanilla', 'dscnn', 'bcresnet')


def parse_args():
    p = argparse.ArgumentParser(description='K-fold single-shot trainer')
    p.add_argument('model', choices=MODEL_CHOICES,
                   help='Which architecture to train.')
    p.add_argument('fold', type=int, choices=(1, 2, 3, 4, 5),
                   help='Fold number (1-indexed).')
    return p.parse_args()


ARGS = parse_args()
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
# phase1_training_utils reads SEED to drive _BASE_SEED + tf.data shuffle seed.
os.environ['SEED'] = str(SEED)

random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


# ---- Paths + imports from src/ ---------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'models'))

from utils.dataset import (  # noqa: E402
    KEYWORDS,
    PROCESSED_DIR,
    ensure_directory,
    load_dataset_for_speakers,
    load_dataset_filtered,
)
from phase1_training_utils import build_phase1_train_datasets  # noqa: E402

FOLD_CONFIG_DIR = PROJECT_ROOT / 'configs' / 'kfold'
KFOLD_MODELS_DIR = PROJECT_ROOT / 'models' / 'kfold'
KFOLD_RESULTS_DIR = PROJECT_ROOT / 'results' / 'kfold'

MODEL_DISPLAY = {
    'vanilla':  'Vanilla',
    'dscnn':    'DS-CNN',
    'bcresnet': 'BC-ResNet',
}


# ---- Data loading -----------------------------------------------------------

def load_fold(fold_n: int):
    cfg_path = FOLD_CONFIG_DIR / f'fold_{fold_n}.json'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'Missing {cfg_path} — run scripts/build_kfold_splits.py first.'
        )
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))

    test_speakers = cfg['test_speakers']
    val_speakers = cfg['val_speakers']
    train_speakers = (
        cfg['train_voicers'] + cfg['train_slr'] + cfg['train_speakers_extra']
    )

    X_train, y_train_lbl, _ = load_dataset_for_speakers(train_speakers)
    X_val, y_val_lbl, _ = load_dataset_filtered(val_speakers, exclude_augmented=True)
    X_test, y_test_lbl, _ = load_dataset_filtered(test_speakers, exclude_augmented=True)

    if len(X_train) == 0:
        raise RuntimeError(f'No train samples loaded for fold {fold_n}')
    if len(X_val) == 0:
        raise RuntimeError(f'No val samples loaded for fold {fold_n}')
    if len(X_test) == 0:
        raise RuntimeError(f'No test samples loaded for fold {fold_n}')

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)
    X_test = X_test[..., np.newaxis].astype(np.float32)

    le = LabelEncoder().fit(KEYWORDS)
    y_train_enc = le.transform(y_train_lbl)
    y_val_enc = le.transform(y_val_lbl)
    y_test_enc = le.transform(y_test_lbl)

    y_train_oh = tf.one_hot(y_train_enc, depth=len(KEYWORDS)).numpy().astype(np.float32)
    y_val_oh = tf.one_hot(y_val_enc, depth=len(KEYWORDS)).numpy().astype(np.float32)

    return {
        'cfg': cfg,
        'train_speakers': train_speakers,
        'val_speakers': val_speakers,
        'test_speakers': test_speakers,
        'X_train': X_train,
        'y_train_oh': y_train_oh,
        'X_val': X_val,
        'y_val_oh': y_val_oh,
        'X_test': X_test,
        'y_test_enc': y_test_enc,
        'label_classes': le.classes_,
    }


# ---- INT8 conversion --------------------------------------------------------

def _representative_dataset(X, max_samples=200):
    n = min(max_samples, len(X))
    # Deterministic subset across the full train array for better coverage
    # than slicing the first n (which would be one speaker's first files).
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X))[:n]
    samples = X[idx].astype(np.float32, copy=False)

    def gen():
        for i in range(len(samples)):
            yield [samples[i:i + 1]]

    return gen


def convert_via_savedmodel(model, X_train, tflite_path: Path):
    saved_dir = tempfile.mkdtemp(prefix='kfold_sm_')
    try:
        model.export(saved_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(X_train)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_bytes = converter.convert()
    finally:
        shutil.rmtree(saved_dir, ignore_errors=True)
    tflite_path.write_bytes(tflite_bytes)
    return tflite_bytes


def convert_qat_keras(qat_model, X_train, tflite_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_quantizer = True
    converter._experimental_disable_per_channel = True
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)
    return tflite_bytes


def evaluate_int8(tflite_path: Path, X_test, y_test_enc, label_names):
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_scale, in_zp = in_det['quantization']
    if in_scale == 0:
        raise RuntimeError('INT8 input scale is 0 — model not properly quantized')

    preds = np.empty(len(X_test), dtype=np.int64)
    for i, sample in enumerate(X_test):
        q = np.round(sample / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_det['index'], q[np.newaxis, ...])
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out_det['index'])))

    acc = accuracy_score(y_test_enc, preds)
    f1m = f1_score(y_test_enc, preds, average='macro')
    report = classification_report(
        y_test_enc, preds,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        digits=4,
        zero_division=0,
    )
    per_class_f1 = f1_score(y_test_enc, preds, average=None,
                            labels=list(range(len(label_names))),
                            zero_division=0)
    return acc, f1m, report, per_class_f1, in_scale, in_zp


# ---- Per-model training pipelines ------------------------------------------

def _phase1_callbacks(checkpoint_path: Path, patience_es=10, patience_lr=5,
                      use_tf_keras=False):
    if use_tf_keras:
        import tf_keras as kb
    else:
        kb = tf.keras
    return [
        kb.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
        ),
        kb.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience_es,
            restore_best_weights=True,
            verbose=0,
        ),
        kb.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            verbose=0,
        ),
    ]


def train_simple(build_fn, data, checkpoint_path):
    """Vanilla CNN / DS-CNN: same Phase 1 recipe, no QAT."""
    model = build_fn(input_shape=(40, 32, 1), num_classes=len(KEYWORDS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1,
        ),
        metrics=['accuracy'],
    )
    train_ds, _ = build_phase1_train_datasets(
        data['X_train'], data['y_train_oh'], batch_size=32, shuffle_buffer=4096,
    )
    callbacks = _phase1_callbacks(checkpoint_path)
    model.fit(
        train_ds,
        validation_data=(data['X_val'], data['y_val_oh']),
        epochs=50,
        callbacks=callbacks,
        verbose=0,
    )
    return model


def train_bcresnet(data, checkpoint_path):
    """BC-ResNet: float Phase 1 pretrain, then 15 epochs of tfmot QAT.

    Mirrors train_bcresnet_phase1.py:run_qat_finetuning — including the
    Keras-3 -> tf_keras fallback when tfmot rejects the Keras-3 functional
    model.
    """
    from models.bc_resnet import build_bc_resnet

    float_model = build_bc_resnet(input_shape=(40, 32, 1), num_classes=len(KEYWORDS))
    float_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1,
        ),
        metrics=['accuracy'],
    )
    train_ds, _ = build_phase1_train_datasets(
        data['X_train'], data['y_train_oh'], batch_size=32, shuffle_buffer=4096,
    )
    float_ckpt = checkpoint_path.parent / f'{checkpoint_path.stem}_float.keras'
    float_model.fit(
        train_ds,
        validation_data=(data['X_val'], data['y_val_oh']),
        epochs=50,
        callbacks=_phase1_callbacks(float_ckpt),
        verbose=0,
    )

    import tensorflow_model_optimization as tfmot
    try:
        qat_model = tfmot.quantization.keras.quantize_model(float_model)
        use_tf_keras = False
    except Exception:
        from train_bcresnet_phase1 import build_bc_resnet_tfkeras
        tfk_model = build_bc_resnet_tfkeras(
            input_shape=(40, 32, 1), num_classes=len(KEYWORDS),
        )
        tfk_model.set_weights(float_model.get_weights())
        qat_model = tfmot.quantization.keras.quantize_model(tfk_model)
        use_tf_keras = True

    if use_tf_keras:
        import tf_keras as kb
    else:
        kb = tf.keras

    qat_model.compile(
        optimizer=kb.optimizers.Adam(learning_rate=1e-4),
        loss=kb.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=0.1,
        ),
        metrics=['accuracy'],
    )
    qat_model.fit(
        train_ds,
        validation_data=(data['X_val'], data['y_val_oh']),
        epochs=15,
        callbacks=_phase1_callbacks(
            checkpoint_path, patience_es=5, patience_lr=3, use_tf_keras=use_tf_keras,
        ),
        verbose=0,
    )
    return qat_model


# ---- Main -------------------------------------------------------------------

def main():
    model_key = ARGS.model
    fold_n = ARGS.fold
    model_display = MODEL_DISPLAY[model_key]

    KFOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    KFOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    data = load_fold(fold_n)

    checkpoint_path = KFOLD_MODELS_DIR / f'{model_key}_fold{fold_n}.keras'
    tflite_path = KFOLD_MODELS_DIR / f'{model_key}_fold{fold_n}_int8.tflite'
    report_path = KFOLD_RESULTS_DIR / f'{model_key}_fold{fold_n}_report.txt'

    if model_key == 'vanilla':
        from vanilla_cnn import build_vanilla_cnn
        model = train_simple(build_vanilla_cnn, data, checkpoint_path)
        is_qat = False
    elif model_key == 'dscnn':
        from ds_cnn import build_ds_cnn
        model = train_simple(build_ds_cnn, data, checkpoint_path)
        is_qat = False
    else:  # bcresnet
        model = train_bcresnet(data, checkpoint_path)
        is_qat = True

    if is_qat:
        tflite_bytes = convert_qat_keras(model, data['X_train'], tflite_path)
    else:
        tflite_bytes = convert_via_savedmodel(model, data['X_train'], tflite_path)
    int8_size_kb = len(tflite_bytes) / 1024.0

    acc, f1m, report, per_class_f1, in_scale, in_zp = evaluate_int8(
        tflite_path, data['X_test'], data['y_test_enc'], data['label_classes'],
    )

    elapsed_min = (time.time() - overall_t0) / 60.0

    # ---- Per-fold report ----------------------------------------------------
    lines = []
    lines.append(f'Model: {model_display}')
    lines.append(f'Fold:  {fold_n}/5')
    lines.append(f'Seed:  {SEED}')
    lines.append('')
    lines.append(f'Train speakers ({len(data["train_speakers"])}): '
                 f'voicers={data["cfg"]["train_voicers"]}; '
                 f'+ {len(data["cfg"]["train_slr"])} slr_* '
                 f'+ {len(data["cfg"]["train_speakers_extra"])} speaker_*')
    lines.append(f'Val   speakers ({len(data["val_speakers"])}): {data["val_speakers"]}')
    lines.append(f'Test  speakers ({len(data["test_speakers"])}): {data["test_speakers"]}')
    lines.append('')
    lines.append(f'Train samples (incl. _aug_):    {len(data["X_train"])}')
    lines.append(f'Val   samples (originals only): {len(data["X_val"])}')
    lines.append(f'Test  samples (originals only): {len(data["X_test"])}')
    lines.append('')
    lines.append(f'INT8 size (KB):       {int8_size_kb:.2f}')
    lines.append(f'INT8 input qparams:   scale={in_scale:.6f} zero_point={in_zp}')
    lines.append(f'INT8 test accuracy:   {acc * 100:.2f}%')
    lines.append(f'INT8 macro F1:        {f1m:.4f}')
    lines.append(f'Training + eval time: {elapsed_min:.2f} min')
    lines.append('')
    lines.append('Per-class INT8 F1')
    lines.append('-' * 40)
    for kw, f1v in zip(data['label_classes'], per_class_f1):
        lines.append(f'  {kw:<12} {f1v:.4f}')
    lines.append('')
    lines.append(report)
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    # Cleanup heavy state before exit (subprocess will release these anyway,
    # but we keep memory peaks bounded if someone runs this in-process).
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    print(f'[{model_display} fold {fold_n}/5] '
          f'INT8 acc: {acc * 100:.2f}%, F1: {f1m:.4f}, '
          f'time: {elapsed_min:.1f} min')


if __name__ == '__main__':
    main()
