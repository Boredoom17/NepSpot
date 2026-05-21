"""BC-ResNet post-training INT8 quantisation (PTQ), seed 42.

The headline BC-ResNet number on disk (84.21% INT8 seed 123, 81.72% +/- 2.20%
across seeds) was produced via QAT (15 extra fine-tuning epochs after the float
pre-QAT checkpoint). Vanilla CNN and DS-CNN reported INT8 numbers use PTQ only.
This script produces a fair PTQ-only BC-ResNet baseline for cross-model
comparison: load the float pre-QAT seed-42 checkpoint, export to SavedModel,
PTQ-convert to INT8 using the same converter flags as the QAT path, evaluate
on the speaker-independent test split (voicer28-30, originals only).

Run:
    cd /Users/ad/codes/NepSpot
    /Users/ad/codes/.venv310/bin/python3 scripts/bcresnet_ptq.py
"""

import contextlib
import io
import os
import shutil
import sys

SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import random
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

from utils.dataset import (  # noqa: E402
    KEYWORDS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_directory,
    load_split_dataset,
    representative_sample_generator,
)


FLOAT_CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_float_best.keras')
SAVED_MODEL_DIR = os.path.join(MODELS_DIR, 'bc_resnet_ptq_seed42_saved_model')
TFLITE_PATH = os.path.join(PROJECT_ROOT, 'models', 'tflite', 'bc_resnet_int8_ptq_seed42.tflite')
REPORT_PATH = os.path.join(RESULTS_DIR, 'metrics', 'bcresnet_ptq_seed42_report.txt')

QAT_REFERENCE_ACC = 84.21  # seed 123 INT8 from seeds_experiment_report.txt
QAT_REFERENCE_SEED = 123


@contextlib.contextmanager
def _silence():
    """Silence both Python-level and C-level stdout/stderr."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)
    try:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        os.close(devnull_fd)


def prepare_test_data():
    X_test, y_test, _, test_speakers = load_split_dataset('test')
    X_test = X_test[..., np.newaxis].astype(np.float32)
    le = LabelEncoder().fit(KEYWORDS)
    y_test_enc = le.transform(y_test)
    return X_test, y_test_enc, le, test_speakers


def load_float_model():
    with _silence():
        model = tf.keras.models.load_model(FLOAT_CHECKPOINT_PATH, compile=False)
    return model


def export_saved_model(model):
    if os.path.exists(SAVED_MODEL_DIR):
        shutil.rmtree(SAVED_MODEL_DIR)
    with _silence():
        model.export(SAVED_MODEL_DIR)


def convert_int8_ptq():
    ensure_directory(os.path.dirname(TFLITE_PATH))
    if os.path.exists(TFLITE_PATH):
        os.remove(TFLITE_PATH)

    with _silence():
        samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i + 1].astype(np.float32)]

    with _silence():
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # Same experimental flags as the QAT->INT8 path; the BC-ResNet broadcast
        # add forces same-scale reconciliation that the new MLIR quantizer
        # handles with per-tensor (not per-channel) scaling.
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_quantizer = True
        converter._experimental_disable_per_channel = True
        tflite_bytes = converter.convert()

    with open(TFLITE_PATH, 'wb') as handle:
        handle.write(tflite_bytes)
    return tflite_bytes


def evaluate_int8(X_test, y_test_enc, label_names):
    with _silence():
        interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
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
    ensure_directory(os.path.dirname(REPORT_PATH))

    X_test, y_test_enc, le, test_speakers = prepare_test_data()

    if not os.path.exists(FLOAT_CHECKPOINT_PATH):
        raise FileNotFoundError(
            'Float pre-QAT BC-ResNet seed-42 checkpoint not found at '
            + FLOAT_CHECKPOINT_PATH
        )

    model = load_float_model()
    print('Float pre-QAT checkpoint loaded from: ' + FLOAT_CHECKPOINT_PATH)

    export_saved_model(model)
    tflite_bytes = convert_int8_ptq()
    int8_size_kb = len(tflite_bytes) / 1024.0
    print('PTQ INT8 conversion complete. Model size: ' + format(int8_size_kb, '.2f') + ' KB')

    acc, macro_f1, classification_text, in_scale, in_zp = evaluate_int8(
        X_test, y_test_enc, le.classes_,
    )
    print('INT8 PTQ test accuracy: ' + format(acc * 100, '.2f') + '%')
    print('INT8 PTQ macro F1: ' + format(macro_f1, '.4f'))
    print(
        'QAT comparison (existing): '
        + format(QAT_REFERENCE_ACC, '.2f') + '% (seed ' + str(QAT_REFERENCE_SEED) + ')'
        + ' — note we are testing seed 42\'s float here for fair comparison'
    )

    header = (
        'BC-ResNet PTQ (post-training INT8 quantisation) - seed 42\n'
        'Reason: fair INT8 comparison vs Vanilla CNN / DS-CNN (PTQ-only).\n'
        '\n'
        'Float pre-QAT checkpoint: '
        + os.path.relpath(FLOAT_CHECKPOINT_PATH, PROJECT_ROOT) + '\n'
        'Seed: ' + str(SEED) + '\n'
        'Evaluation split: speaker-independent test (originals only, no _aug_)\n'
        'Test speakers: ' + ', '.join(test_speakers) + '\n'
        'Test samples: ' + str(len(X_test)) + '\n'
        '\n'
        'PTQ converter settings:\n'
        '  optimizations:                            [tf.lite.Optimize.DEFAULT]\n'
        '  representative_dataset:                   200 train samples (same as QAT path)\n'
        '  target_spec.supported_ops:                [TFLITE_BUILTINS_INT8]\n'
        '  inference_input_type / output_type:       int8 / int8\n'
        '  experimental_enable_resource_variables:   True\n'
        '  experimental_new_quantizer:               True\n'
        '  _experimental_disable_per_channel:        True\n'
        '\n'
        'INT8 model size (KB):           ' + format(int8_size_kb, '.2f') + '\n'
        'INT8 input qparams:             scale=' + format(in_scale, '.6f')
        + ' zero_point=' + str(in_zp) + '\n'
        'INT8 PTQ test accuracy:         ' + format(acc * 100, '.2f') + '%\n'
        'INT8 PTQ macro F1:              ' + format(macro_f1, '.4f') + '\n'
        '\n'
        'TFLite output: ' + os.path.relpath(TFLITE_PATH, PROJECT_ROOT) + '\n'
        '\n'
        'Cross-method reference for the same float-42 checkpoint:\n'
        '  QAT INT8 (different seeds, in-train pipeline):\n'
        '    seed 42  : 84.76% / F1 0.847 (h5-weight-transferred recovered)\n'
        '    seed 123 : 84.21% / F1 0.839\n'
        '    seed 456 : 80.33% / F1 0.805\n'
        '    N=3 mean : 81.72% +/- 2.20%\n'
        '  Vanilla CNN INT8 PTQ (N=3 mean): 83.75% +/- 1.12% (F1 0.834)\n'
        '  DS-CNN     INT8 PTQ (N=3 mean): 79.04% +/- 1.15% (F1 0.785)\n'
        '\n'
        'Per-class classification report (INT8 PTQ):\n'
    )

    with open(REPORT_PATH, 'w', encoding='utf-8') as handle:
        handle.write(header)
        handle.write(classification_text)
        handle.write('\n')


if __name__ == '__main__':
    main()
