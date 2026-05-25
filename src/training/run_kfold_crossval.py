"""5-fold speaker-independent cross-validation for NepSpot.

Trains Vanilla CNN, DS-CNN, and BC-ResNet on 5 different speaker splits of the
voicer pool, using the same Phase 1 training recipe (SpecAugment + Mixup +
label smoothing; QAT fine-tune for BC-ResNet) as the single-split scripts.

Run from the project root:
    source /Users/ad/codes/.venv310/bin/activate
    cd /Users/ad/codes/NepSpot
    python src/models/run_kfold_crossval.py
"""

import os
import random
import re
import shutil
import sys
import time
from typing import Dict, List, Sequence, Tuple

os.environ['PYTHONHASHSEED'] = '42'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(42)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

from vanilla_cnn import build_vanilla_cnn
from ds_cnn import build_ds_cnn
from bc_resnet import build_bc_resnet
from _training_utils import build_phase1_train_datasets, inspect_augmentation_behavior
from data.dataset import (
    KEYWORDS,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ensure_directory,
    load_dataset_for_speakers,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TFLITE_DIR = os.path.join(PROJECT_ROOT, 'models', 'tflite')
KFOLD_REPORTS_DIR = os.path.join(RESULTS_DIR, 'metrics', 'kfold')
KFOLD_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'metrics', 'kfold_summary.txt')

NUM_FOLDS = 5
TEST_PER_FOLD = 5
VAL_PER_FOLD = 3
PHASE1_EPOCHS = 50
QAT_EPOCHS = 15
BATCH_SIZE = 32
INPUT_SHAPE = (40, 32, 1)


# ---------------------------------------------------------------------------
# Speaker discovery & fold assignment
# ---------------------------------------------------------------------------


def discover_voicers() -> List[str]:
    """Return the voicerN folders that exist in data/processed, sorted by N."""
    pattern = re.compile(r'^voicer(\d+)$')
    found = []
    for name in os.listdir(PROCESSED_DIR):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(PROCESSED_DIR, name)):
            found.append((int(match.group(1)), name))
    found.sort()
    return [name for _, name in found]


def discover_external_train_speakers() -> List[str]:
    """speaker_1..N and slr_NNN folders are always in train across all folds."""
    speakers = []
    for name in sorted(os.listdir(PROCESSED_DIR)):
        path = os.path.join(PROCESSED_DIR, name)
        if not os.path.isdir(path):
            continue
        if name.startswith('speaker_') or name.startswith('slr_'):
            speakers.append(name)
    speakers.sort(key=lambda n: (n.split('_')[0], int(n.split('_')[1])))
    return speakers


def build_folds(voicers: Sequence[str], num_folds: int = NUM_FOLDS,
                test_per_fold: int = TEST_PER_FOLD,
                val_per_fold: int = VAL_PER_FOLD) -> List[Dict[str, List[str]]]:
    """Partition voicers into N folds: each fold rotates test, val, train slices."""
    n = len(voicers)
    if n < test_per_fold + val_per_fold + 1:
        raise ValueError(f'Need at least {test_per_fold + val_per_fold + 1} voicers, found {n}')

    folds = []
    for fold_idx in range(num_folds):
        test_start = (fold_idx * test_per_fold) % n
        test_speakers = [voicers[(test_start + i) % n] for i in range(test_per_fold)]

        val_start = (test_start + test_per_fold) % n
        val_speakers = [voicers[(val_start + i) % n] for i in range(val_per_fold)]

        used = set(test_speakers) | set(val_speakers)
        train_voicers = [v for v in voicers if v not in used]

        folds.append({
            'fold_id': fold_idx + 1,
            'test': test_speakers,
            'val': val_speakers,
            'train_voicers': train_voicers,
        })
    return folds


# ---------------------------------------------------------------------------
# Data loading for a fold
# ---------------------------------------------------------------------------


def load_fold_data(fold: Dict[str, List[str]], external_speakers: Sequence[str]):
    train_speakers = list(fold['train_voicers']) + list(external_speakers)
    val_speakers = list(fold['val'])
    test_speakers = list(fold['test'])
    assert len(KEYWORDS) == 12, f'Expected 12 keywords, got {len(KEYWORDS)}'

    print(f"  Loading train ({len(train_speakers)} speaker folders)...")
    X_train, y_train, _ = load_dataset_for_speakers(train_speakers, PROCESSED_DIR, KEYWORDS)
    print(f"  Loading val   ({len(val_speakers)} speakers)...")
    X_val, y_val, _ = load_dataset_for_speakers(val_speakers, PROCESSED_DIR, KEYWORDS)
    print(f"  Loading test  ({len(test_speakers)} speaker folders)...")
    X_test, y_test, _ = load_dataset_for_speakers(test_speakers, PROCESSED_DIR, KEYWORDS)

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)
    X_test = X_test[..., np.newaxis].astype(np.float32)

    le = LabelEncoder().fit(KEYWORDS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    y_train_oh = tf.one_hot(y_train_enc, depth=len(KEYWORDS)).numpy()
    y_val_oh = tf.one_hot(y_val_enc, depth=len(KEYWORDS)).numpy()

    print(f"  Train: {len(X_train)} samples | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        'X_train': X_train, 'y_train_oh': y_train_oh, 'y_train_enc': y_train_enc,
        'X_val': X_val, 'y_val_oh': y_val_oh, 'y_val_enc': y_val_enc,
        'X_test': X_test, 'y_test_enc': y_test_enc,
        'label_encoder': le,
        'train_speakers': train_speakers,
        'val_speakers': val_speakers,
        'test_speakers': test_speakers,
    }


# ---------------------------------------------------------------------------
# Training building blocks
# ---------------------------------------------------------------------------


def reset_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_phase1(model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=['accuracy'],
    )


def make_phase1_callbacks(checkpoint_path: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]


def train_phase1(model, fold_data, checkpoint_path: str, epochs: int = PHASE1_EPOCHS) -> None:
    ensure_directory(os.path.dirname(checkpoint_path))
    train_ds, debug_ds = build_phase1_train_datasets(
        fold_data['X_train'], fold_data['y_train_oh'],
        batch_size=BATCH_SIZE, shuffle_buffer=4096,
    )
    stats = inspect_augmentation_behavior(debug_ds, max_batches=12)
    print(f"  Augment probe: specaug={stats['specaug_examples']} "
          f"mixup_batches={stats['mixup_batches']}/{stats['observed_batches']}")

    model.fit(
        train_ds,
        validation_data=(fold_data['X_val'], fold_data['y_val_oh']),
        epochs=epochs,
        callbacks=make_phase1_callbacks(checkpoint_path),
        verbose=1,
    )


# ---------------------------------------------------------------------------
# BC-ResNet QAT (mirrors train_bcresnet_phase1.run_qat_finetuning)
# ---------------------------------------------------------------------------


def run_qat_finetuning(float_model, fold_data, qat_checkpoint_path: str,
                       epochs: int = QAT_EPOCHS):
    import tensorflow_model_optimization as tfmot

    try:
        qat_model = tfmot.quantization.keras.quantize_model(float_model)
    except Exception:
        from train_bcresnet_phase1 import build_bc_resnet_tfkeras
        tfk_model = build_bc_resnet_tfkeras(input_shape=INPUT_SHAPE, num_classes=len(KEYWORDS))
        tfk_model.set_weights(float_model.get_weights())
        qat_model = tfmot.quantization.keras.quantize_model(tfk_model)

    if qat_model.__class__.__module__.startswith('tf_keras'):
        import tf_keras as keras_backend
    else:
        keras_backend = tf.keras

    qat_model.compile(
        optimizer=keras_backend.optimizers.Adam(learning_rate=1e-4),
        loss=keras_backend.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=['accuracy'],
    )

    ensure_directory(os.path.dirname(qat_checkpoint_path))
    callbacks = [
        keras_backend.callbacks.ModelCheckpoint(
            filepath=qat_checkpoint_path, monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
        keras_backend.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1,
        ),
        keras_backend.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1,
        ),
    ]

    train_ds, _ = build_phase1_train_datasets(
        fold_data['X_train'], fold_data['y_train_oh'],
        batch_size=BATCH_SIZE, shuffle_buffer=4096,
    )
    qat_model.fit(
        train_ds,
        validation_data=(fold_data['X_val'], fold_data['y_val_oh']),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return qat_model


# ---------------------------------------------------------------------------
# INT8 conversion & evaluation
# ---------------------------------------------------------------------------


def fold_representative_dataset(X_train: np.ndarray, max_samples: int = 200):
    n = min(max_samples, len(X_train))
    indices = np.random.default_rng(seed=42).permutation(len(X_train))[:n]
    chosen = X_train[indices].astype(np.float32)

    def generator():
        for i in range(len(chosen)):
            yield [chosen[i:i + 1]]
    return generator


def convert_savedmodel_to_int8(saved_model_path: str, tflite_path: str,
                               X_train: np.ndarray) -> bytes:
    ensure_directory(os.path.dirname(tflite_path))
    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = fold_representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    with open(tflite_path, 'wb') as handle:
        handle.write(tflite_bytes)
    return tflite_bytes


def convert_qat_keras_to_int8(qat_model, tflite_path: str,
                              X_train: np.ndarray) -> bytes:
    ensure_directory(os.path.dirname(tflite_path))
    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = fold_representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    with open(tflite_path, 'wb') as handle:
        handle.write(tflite_bytes)
    return tflite_bytes


def evaluate_int8_tflite(tflite_path: str, X_test: np.ndarray,
                         y_test_enc: np.ndarray) -> Tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    if input_scale == 0:
        raise RuntimeError('Invalid INT8 quantization scale')

    predictions = []
    for sample in X_test:
        q = np.round(sample / input_scale + input_zero_point).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], q[np.newaxis, ...])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(int(np.argmax(out)))
    preds = np.array(predictions)
    return accuracy_score(y_test_enc, preds), f1_score(y_test_enc, preds, average='macro')


def evaluate_keras(model, X_test, y_test_enc, label_names):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test_enc, y_pred)
    macro_f1 = f1_score(y_test_enc, y_pred, average='macro')
    report = classification_report(
        y_test_enc, y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    return acc, macro_f1, report


# ---------------------------------------------------------------------------
# Per-model training drivers
# ---------------------------------------------------------------------------


def export_saved_model(model, saved_model_path: str) -> None:
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)
    model.export(saved_model_path)


def write_fold_report(report_path: str, model_name: str, fold_id: int,
                      fold_data, float_acc: float, float_f1: float,
                      int8_acc: float, int8_f1: float, int8_size_kb: float,
                      classification_text: str, train_seconds: float) -> None:
    ensure_directory(os.path.dirname(report_path))
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(f'Model: {model_name}\n')
        handle.write(f'Fold: {fold_id}/{NUM_FOLDS}\n')
        handle.write(f'Train wall time: {train_seconds:.1f} s\n\n')
        handle.write(f'Train speakers ({len(fold_data["train_speakers"])}): '
                     f'{", ".join(fold_data["train_speakers"][:20])}'
                     f'{"..." if len(fold_data["train_speakers"]) > 20 else ""}\n')
        handle.write(f'Val speakers: {", ".join(fold_data["val_speakers"])}\n')
        handle.write(f'Test speakers: {", ".join(fold_data["test_speakers"])}\n')
        handle.write(f'Train samples: {len(fold_data["X_train"])}\n')
        handle.write(f'Val samples:   {len(fold_data["X_val"])}\n')
        handle.write(f'Test samples:  {len(fold_data["X_test"])}\n\n')
        handle.write(f'Float32 test accuracy: {float_acc * 100:.2f}%\n')
        handle.write(f'Float32 macro F1:      {float_f1:.4f}\n')
        handle.write(f'INT8 test accuracy:    {int8_acc * 100:.2f}%\n')
        handle.write(f'INT8 macro F1:         {int8_f1:.4f}\n')
        handle.write(f'INT8 model size (KB):  {int8_size_kb:.2f}\n\n')
        handle.write(classification_text)


def train_vanilla_cnn_fold(fold_data, fold_id: int) -> Dict[str, float]:
    print(f"\n[Vanilla CNN | fold {fold_id}] Building model...")
    reset_seeds(42)
    model = build_vanilla_cnn(input_shape=INPUT_SHAPE, num_classes=len(KEYWORDS))
    compile_phase1(model)

    best_keras = os.path.join(MODELS_DIR, f'vanilla_cnn_fold{fold_id}_best.keras')
    saved_model_path = os.path.join(MODELS_DIR, f'vanilla_cnn_fold{fold_id}_savedmodel')
    tflite_path = os.path.join(TFLITE_DIR, f'vanilla_cnn_int8_fold{fold_id}.tflite')
    report_path = os.path.join(KFOLD_REPORTS_DIR, f'vanilla_cnn_fold{fold_id}_report.txt')

    t0 = time.time()
    train_phase1(model, fold_data, best_keras)
    train_seconds = time.time() - t0

    float_acc, float_f1, cls_text = evaluate_keras(
        model, fold_data['X_test'], fold_data['y_test_enc'], sorted(KEYWORDS),
    )
    export_saved_model(model, saved_model_path)
    tflite_bytes = convert_savedmodel_to_int8(saved_model_path, tflite_path, fold_data['X_train'])
    int8_acc, int8_f1 = evaluate_int8_tflite(tflite_path, fold_data['X_test'], fold_data['y_test_enc'])

    write_fold_report(report_path, 'Vanilla CNN', fold_id, fold_data,
                      float_acc, float_f1, int8_acc, int8_f1,
                      len(tflite_bytes) / 1024.0, cls_text, train_seconds)
    print(f"[Vanilla CNN | fold {fold_id}] float32={float_acc*100:.2f}% "
          f"int8={int8_acc*100:.2f}% (wall {train_seconds/60:.1f} min)")

    tf.keras.backend.clear_session()
    return {'float_acc': float_acc, 'float_f1': float_f1,
            'int8_acc': int8_acc, 'int8_f1': int8_f1, 'train_seconds': train_seconds}


def train_ds_cnn_fold(fold_data, fold_id: int) -> Dict[str, float]:
    print(f"\n[DS-CNN | fold {fold_id}] Building model...")
    reset_seeds(42)
    model = build_ds_cnn(input_shape=INPUT_SHAPE, num_classes=len(KEYWORDS))
    compile_phase1(model)

    best_keras = os.path.join(MODELS_DIR, f'ds_cnn_fold{fold_id}_best.keras')
    saved_model_path = os.path.join(MODELS_DIR, f'ds_cnn_fold{fold_id}_savedmodel')
    tflite_path = os.path.join(TFLITE_DIR, f'ds_cnn_int8_fold{fold_id}.tflite')
    report_path = os.path.join(KFOLD_REPORTS_DIR, f'ds_cnn_fold{fold_id}_report.txt')

    t0 = time.time()
    train_phase1(model, fold_data, best_keras)
    train_seconds = time.time() - t0

    float_acc, float_f1, cls_text = evaluate_keras(
        model, fold_data['X_test'], fold_data['y_test_enc'], sorted(KEYWORDS),
    )
    export_saved_model(model, saved_model_path)
    tflite_bytes = convert_savedmodel_to_int8(saved_model_path, tflite_path, fold_data['X_train'])
    int8_acc, int8_f1 = evaluate_int8_tflite(tflite_path, fold_data['X_test'], fold_data['y_test_enc'])

    write_fold_report(report_path, 'DS-CNN', fold_id, fold_data,
                      float_acc, float_f1, int8_acc, int8_f1,
                      len(tflite_bytes) / 1024.0, cls_text, train_seconds)
    print(f"[DS-CNN | fold {fold_id}] float32={float_acc*100:.2f}% "
          f"int8={int8_acc*100:.2f}% (wall {train_seconds/60:.1f} min)")

    tf.keras.backend.clear_session()
    return {'float_acc': float_acc, 'float_f1': float_f1,
            'int8_acc': int8_acc, 'int8_f1': int8_f1, 'train_seconds': train_seconds}


def train_bc_resnet_fold(fold_data, fold_id: int) -> Dict[str, float]:
    print(f"\n[BC-ResNet | fold {fold_id}] Building model...")
    reset_seeds(42)
    float_model = build_bc_resnet(input_shape=INPUT_SHAPE, num_classes=len(KEYWORDS))
    compile_phase1(float_model)

    float_ckpt = os.path.join(MODELS_DIR, f'bc_resnet_fold{fold_id}_float_best.keras')
    qat_ckpt = os.path.join(MODELS_DIR, f'bc_resnet_fold{fold_id}_best.keras')
    saved_model_path = os.path.join(MODELS_DIR, f'bc_resnet_fold{fold_id}_savedmodel')
    tflite_path = os.path.join(TFLITE_DIR, f'bc_resnet_int8_fold{fold_id}.tflite')
    report_path = os.path.join(KFOLD_REPORTS_DIR, f'bc_resnet_fold{fold_id}_report.txt')

    t0 = time.time()
    train_phase1(float_model, fold_data, float_ckpt, epochs=PHASE1_EPOCHS)
    print(f"[BC-ResNet | fold {fold_id}] Starting QAT fine-tune...")
    qat_model = run_qat_finetuning(float_model, fold_data, qat_ckpt, epochs=QAT_EPOCHS)
    train_seconds = time.time() - t0

    float_acc, float_f1, cls_text = evaluate_keras(
        qat_model, fold_data['X_test'], fold_data['y_test_enc'], sorted(KEYWORDS),
    )
    export_saved_model(qat_model, saved_model_path)
    tflite_bytes = convert_qat_keras_to_int8(qat_model, tflite_path, fold_data['X_train'])
    int8_acc, int8_f1 = evaluate_int8_tflite(tflite_path, fold_data['X_test'], fold_data['y_test_enc'])

    write_fold_report(report_path, 'BC-ResNet', fold_id, fold_data,
                      float_acc, float_f1, int8_acc, int8_f1,
                      len(tflite_bytes) / 1024.0, cls_text, train_seconds)
    print(f"[BC-ResNet | fold {fold_id}] float32={float_acc*100:.2f}% "
          f"int8={int8_acc*100:.2f}% (wall {train_seconds/60:.1f} min)")

    tf.keras.backend.clear_session()
    return {'float_acc': float_acc, 'float_f1': float_f1,
            'int8_acc': int8_acc, 'int8_f1': int8_f1, 'train_seconds': train_seconds}


MODEL_DRIVERS = [
    ('Vanilla CNN', train_vanilla_cnn_fold),
    ('DS-CNN', train_ds_cnn_fold),
    ('BC-ResNet', train_bc_resnet_fold),
]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_summary(all_results: Dict[str, List[Dict[str, float]]],
                  folds: Sequence[Dict[str, List[str]]]) -> None:
    ensure_directory(os.path.dirname(KFOLD_SUMMARY_PATH))
    lines = []
    lines.append(f'{NUM_FOLDS}-Fold Cross-Validation Results')
    lines.append('=' * 50)
    lines.append('')
    lines.append('Fold compositions:')
    for fold in folds:
        lines.append(f"  Fold {fold['fold_id']}: test={','.join(fold['test'])} | "
                     f"val={','.join(fold['val'])}")
    lines.append('')

    for model_name, _ in MODEL_DRIVERS:
        results = all_results.get(model_name, [])
        if not results:
            continue
        lines.append(f'{model_name}:')
        for i, r in enumerate(results, start=1):
            lines.append(f"  Fold {i}: float32 {r['float_acc']*100:.2f}%  "
                         f"int8 {r['int8_acc']*100:.2f}%  macroF1 {r['float_f1']:.4f}")
        f_mean, f_std = _mean_std([r['float_acc'] for r in results])
        q_mean, q_std = _mean_std([r['int8_acc'] for r in results])
        m_mean, m_std = _mean_std([r['float_f1'] for r in results])
        lines.append(f"  Mean float32: {f_mean*100:.2f}% ± {f_std*100:.2f}%")
        lines.append(f"  Mean int8:    {q_mean*100:.2f}% ± {q_std*100:.2f}%")
        lines.append(f"  Mean macroF1: {m_mean:.4f} ± {m_std:.4f}")
        lines.append('')

    text = '\n'.join(lines)
    print('\n' + text)
    with open(KFOLD_SUMMARY_PATH, 'w', encoding='utf-8') as handle:
        handle.write(text + '\n')
    print(f'\nSummary written to: {KFOLD_SUMMARY_PATH}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_directory(MODELS_DIR)
    ensure_directory(TFLITE_DIR)
    ensure_directory(KFOLD_REPORTS_DIR)

    voicers = discover_voicers()
    external = discover_external_train_speakers()

    print(f'Discovered {len(voicers)} voicers: {", ".join(voicers)}')
    print(f'External train speakers (constant across folds): {len(external)} '
          f'(speaker_*, slr_*)')
    print(f'12-class mode: silence/unknown speakers excluded from all splits')

    folds = build_folds(voicers)
    print('\nFold layout:')
    for f in folds:
        print(f"  fold {f['fold_id']}: test={f['test']} | val={f['val']} "
              f"| train_voicers={len(f['train_voicers'])}")

    all_results: Dict[str, List[Dict[str, float]]] = {name: [] for name, _ in MODEL_DRIVERS}

    overall_t0 = time.time()
    for fold in folds:
        fold_id = fold['fold_id']
        print('\n' + '=' * 70)
        print(f'FOLD {fold_id}/{NUM_FOLDS}')
        print('=' * 70)
        fold_data = load_fold_data(fold, external)

        for model_name, driver in MODEL_DRIVERS:
            metrics = driver(fold_data, fold_id)
            all_results[model_name].append(metrics)
            write_summary(all_results, folds)

    total_minutes = (time.time() - overall_t0) / 60.0
    print(f'\nAll folds complete. Total wall time: {total_minutes:.1f} min')
    write_summary(all_results, folds)


if __name__ == '__main__':
    main()
