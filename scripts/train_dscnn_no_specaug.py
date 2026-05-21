"""DS-CNN ablation — no SpecAugment, no Mixup. Single seed=42.

Mirrors the Phase 1 DS-CNN training recipe (build_ds_cnn architecture, Adam
1e-3, categorical CE with 0.1 label smoothing, ModelCheckpoint + EarlyStopping
on val_accuracy with patience 10, ReduceLROnPlateau on val_loss with patience
5, 50 max epochs, PTQ INT8 via SavedModel) but replaces the
phase1_training_utils SpecAug+Mixup pipeline with a plain
from_tensor_slices -> shuffle -> batch -> prefetch pipeline. The point of
the run is to measure the contribution of SpecAug+Mixup to the
79.04% +/- 1.15% INT8 baseline.

Run:
    cd /Users/ad/codes/NepSpot
    /Users/ad/codes/.venv310/bin/python3 scripts/train_dscnn_no_specaug.py
"""

import os
import random
import shutil
import sys
import time

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'models'))

from ds_cnn import build_ds_cnn  # noqa: E402
from utils.dataset import (  # noqa: E402
    KEYWORDS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_directory,
    load_split_dataset,
    representative_sample_generator,
)


TFLITE_PATH = os.path.join(PROJECT_ROOT, 'models', 'tflite', 'dscnn_no_specaug_int8.tflite')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics', 'dscnn_no_specaug_report.txt')
BEST_KERAS_PATH = os.path.join(MODELS_DIR, 'dscnn_no_specaug_best.keras')
SAVED_MODEL_DIR = os.path.join(MODELS_DIR, 'dscnn_no_specaug_saved_model')


def prepare_data():
    X_train, y_train, _, _ = load_split_dataset('train')
    X_val, y_val, _, _ = load_split_dataset('val')
    X_test, y_test, _, test_speakers = load_split_dataset('test')

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)
    X_test = X_test[..., np.newaxis].astype(np.float32)

    le = LabelEncoder().fit(KEYWORDS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    y_train_oh = tf.one_hot(y_train_enc, depth=len(KEYWORDS)).numpy().astype(np.float32)
    y_val_oh = tf.one_hot(y_val_enc, depth=len(KEYWORDS)).numpy().astype(np.float32)

    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_enc, le, test_speakers


def build_simple_dataset(X_train, y_train_oh, batch_size=32, shuffle_buffer=4096):
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
    ds = ds.shuffle(min(shuffle_buffer, len(X_train)), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds


def quantize_to_int8(saved_model_path, tflite_path):
    ensure_directory(os.path.dirname(tflite_path))
    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_bytes)
    return tflite_bytes


def evaluate_int8(tflite_path, X_test, y_test_enc, label_names):
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_scale, in_zp = in_det[0]['quantization']
    if in_scale == 0:
        raise RuntimeError('Invalid INT8 input scale (0)')

    preds = []
    for sample in X_test:
        q = np.round(sample / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(in_det[0]['index'], q[np.newaxis, ...])
        interp.invoke()
        preds.append(int(np.argmax(interp.get_tensor(out_det[0]['index']))))
    preds = np.array(preds)

    acc = accuracy_score(y_test_enc, preds)
    macro_f1 = f1_score(y_test_enc, preds, average='macro')
    report = classification_report(
        y_test_enc, preds,
        labels=list(range(len(label_names))),
        target_names=list(label_names),
        digits=4,
        zero_division=0,
    )
    return acc, macro_f1, report, in_scale, in_zp


def main():
    ensure_directory(MODELS_DIR)
    ensure_directory(os.path.dirname(TFLITE_PATH))
    ensure_directory(os.path.dirname(REPORT_PATH))

    X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_enc, le, test_speakers = prepare_data()

    train_ds = build_simple_dataset(X_train, y_train_oh, batch_size=32, shuffle_buffer=4096)

    model = build_ds_cnn(input_shape=(40, 32, 1), num_classes=len(KEYWORDS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_KERAS_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=0,
        ),
    ]

    t0 = time.time()
    model.fit(
        train_ds,
        validation_data=(X_val, y_val_oh),
        epochs=50,
        callbacks=callbacks,
        verbose=0,
    )
    train_minutes = (time.time() - t0) / 60.0

    if os.path.exists(SAVED_MODEL_DIR):
        shutil.rmtree(SAVED_MODEL_DIR)
    model.export(SAVED_MODEL_DIR)

    tflite_bytes = quantize_to_int8(SAVED_MODEL_DIR, TFLITE_PATH)
    int8_size_kb = len(tflite_bytes) / 1024.0

    acc, macro_f1, report, in_scale, in_zp = evaluate_int8(
        TFLITE_PATH, X_test, y_test_enc, le.classes_,
    )

    header = (
        'DS-CNN ablation — NO SpecAugment, NO Mixup\n'
        f'Seed: {SEED}\n'
        f'Speakers (test): {", ".join(test_speakers)}\n'
        f'Test samples: {len(X_test)} (voicer28-30 originals only)\n'
        f'Train pipeline: from_tensor_slices -> shuffle(4096, seed={SEED}) -> batch(32) -> prefetch(1)\n'
        f'Training time:  {train_minutes:.1f} min\n'
        f'INT8 model size (KB): {int8_size_kb:.2f}\n'
        f'INT8 input qparams:   scale={in_scale:.6f} zero_point={in_zp}\n'
        f'INT8 test accuracy:   {acc * 100:.2f}%\n'
        f'INT8 macro F1:        {macro_f1:.4f}\n'
        '\n'
        f'TFLite output: {os.path.relpath(TFLITE_PATH, PROJECT_ROOT)}\n'
        '\n'
        'Baseline reference (DS-CNN with SpecAug+Mixup, N=3 seeds):\n'
        '  INT8 test accuracy: 79.04% +/- 1.15%\n'
        '\n'
    )

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(report)
        f.write('\n')

    print(header + report)


if __name__ == '__main__':
    main()
