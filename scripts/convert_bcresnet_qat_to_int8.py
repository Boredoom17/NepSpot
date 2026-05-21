"""Convert the existing BC-ResNet-1 QAT checkpoint to INT8 TFLite.

Re-runs ONLY the INT8 conversion + evaluation step on the QAT checkpoint
already on disk at models/saved/bc_resnet_phase1_best.keras. No retraining.

Usage:
    cd /Users/ad/codes/NepSpot
    /Users/ad/codes/.venv310/bin/python3 scripts/convert_bcresnet_qat_to_int8.py
"""

import os

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'models'))

from utils.dataset import (  # noqa: E402
    KEYWORDS,
    MODELS_DIR,
    ensure_directory,
    load_split_dataset,
    representative_sample_generator,
)

QAT_CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_best.keras')
TFLITE_DIR = os.path.join(PROJECT_ROOT, 'models', 'tflite')
INT8_TFLITE_PATH = os.path.join(TFLITE_DIR, 'bc_resnet_int8_phase1.tflite')
INT8_HEADER_PATH = os.path.join(TFLITE_DIR, 'bc_resnet_int8_phase1.h')


def load_qat_model(path):
    """Load the QAT-wrapped tf_keras model saved by train_bcresnet_phase1.py."""
    import tf_keras
    import tensorflow_model_optimization as tfmot

    if not os.path.exists(path):
        raise FileNotFoundError(f'QAT checkpoint not found at {path}')

    with tfmot.quantization.keras.quantize_scope():
        model = tf_keras.models.load_model(path, compile=False)
    return model


def convert_to_int8(qat_model, tflite_path):
    ensure_directory(os.path.dirname(tflite_path))
    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Required to convert BC-ResNet's broadcast path through tfmot QAT (see
    # train_bcresnet_phase1.quantize_qat_model_to_int8 for the explanation).
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_quantizer = True
    converter._experimental_disable_per_channel = True

    tflite_bytes = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_bytes)
    return tflite_bytes


def evaluate_int8(tflite_path, X_test, y_test_enc, label_names):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zp = input_details[0]['quantization']
    if in_scale == 0:
        raise RuntimeError('Invalid INT8 input scale (0)')

    preds = []
    for sample in X_test:
        q = np.round(sample / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], q[np.newaxis, ...])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        preds.append(int(np.argmax(out)))
    preds = np.array(preds)

    acc = accuracy_score(y_test_enc, preds)
    macro_f1 = f1_score(y_test_enc, preds, average='macro')
    report = classification_report(
        y_test_enc, preds,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
    )
    return acc, macro_f1, report, in_scale, in_zp


def write_tflite_header(tflite_path, header_path, array_name='nepspot_model_data'):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    include_guard = f'{array_name.upper()}_H_'
    hex_bytes = [f'0x{b:02x}' for b in data]
    body = ',\n'.join(
        '  ' + ', '.join(hex_bytes[i:i + 12])
        for i in range(0, len(hex_bytes), 12)
    )
    content = (
        f'#ifndef {include_guard}\n'
        f'#define {include_guard}\n\n'
        '#include <stdint.h>\n\n'
        f'const unsigned char {array_name}[] = {{\n'
        f'{body}\n'
        '};\n\n'
        f'const unsigned int {array_name}_len = {len(data)};\n\n'
        f'#endif  // {include_guard}\n'
    )
    with open(header_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    print(f'Loading QAT checkpoint: {QAT_CHECKPOINT_PATH}')
    qat_model = load_qat_model(QAT_CHECKPOINT_PATH)
    n_weights = sum(int(np.prod(v.shape)) for v in qat_model.weights)
    print(f'  Loaded QAT model: {len(qat_model.weights)} weight tensors, '
          f'{n_weights:,} total elements')

    print('Loading speaker-independent test split (voicer28-30)...')
    X_test, y_test, _, test_speakers = load_split_dataset('test')
    X_test = X_test[..., np.newaxis].astype(np.float32)

    le = LabelEncoder().fit(KEYWORDS)
    y_test_enc = le.transform(y_test)
    print(f'  Test speakers: {test_speakers}')
    print(f'  Test samples:  {len(X_test)}')

    print(f'\nConverting QAT -> INT8 TFLite')
    print(f'  Output: {INT8_TFLITE_PATH}')
    tflite_bytes = convert_to_int8(qat_model, INT8_TFLITE_PATH)
    size_kb = len(tflite_bytes) / 1024.0
    print(f'  Wrote {size_kb:.2f} KB')

    print('\nEvaluating INT8 TFLite on test set...')
    acc, macro_f1, report, in_scale, in_zp = evaluate_int8(
        INT8_TFLITE_PATH, X_test, y_test_enc, le.classes_,
    )
    print(f'  INT8 input quantization: scale={in_scale:.6f}, zero_point={in_zp}')
    print(f'  INT8 test accuracy: {acc * 100:.2f}%')
    print(f'  INT8 macro F1:      {macro_f1:.4f}')
    print()
    print(report)

    print(f'Writing Arduino header: {INT8_HEADER_PATH}')
    write_tflite_header(INT8_TFLITE_PATH, INT8_HEADER_PATH, array_name='nepspot_model_data')
    print('Done.')


if __name__ == '__main__':
    main()
