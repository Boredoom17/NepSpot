"""Recover INT8 TFLite from the existing BC-ResNet QAT checkpoint.

The QAT checkpoint at models/saved/bc_resnet_phase1_best.keras was trained with
the old AvgPool+UpSampling broadcast. tf_keras.models.load_model fails on it
("Layer 'quantize_layer' expected 5 variables, but received 3"), so this script
bypasses the model loader entirely:

  1. Open the .keras file as a ZIP archive.
  2. Extract model.weights.h5 in-memory.
  3. Walk config.json to enumerate QuantizeWrapperV2 layers in construction
     order and map each to its h5 group (quantize_wrapper_v2, _1, _2, ...).
  4. Pull inner-layer weights out of each wrapper by tfmot's storage convention:
       - Conv2D / DepthwiseConv2D (no bias): kernel at vars/0
       - Dense:                              kernel at vars/0, bias at vars/1
       - BatchNormalization:                 [gamma, beta, mean, var] at layer/vars/0..3
  5. Build the new architecture (DepthwiseConv broadcast variant) and assign
     weights by inner layer name. The new *_freq_avg_broadcast layers have no
     QAT counterpart and keep their frozen 1/F init.
  6. PTQ-convert to INT8 with 200 representative samples from the train split.
  7. Evaluate INT8 on the speaker-independent test split (voicer28-30 originals
     — no _aug_ files exist in those folders) and write the report.
"""

import os
import sys

os.environ.setdefault('PYTHONHASHSEED', '42')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import io
import json
import zipfile
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from utils.dataset import (  # noqa: E402
    KEYWORDS,
    MODELS_DIR,
    ensure_directory,
    load_split_dataset,
    representative_sample_generator,
)
from models.bc_resnet import build_bc_resnet  # noqa: E402

QAT_CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_best.keras')
TFLITE_DIR = os.path.join(PROJECT_ROOT, 'models', 'tflite')
METRICS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
INT8_TFLITE_PATH = os.path.join(TFLITE_DIR, 'bc_resnet_int8_recovered.tflite')
REPORT_PATH = os.path.join(METRICS_DIR, 'bc_resnet_int8_recovered_report.txt')


def _wrapper_h5_name(idx: int) -> str:
    return 'quantize_wrapper_v2' if idx == 0 else f'quantize_wrapper_v2_{idx}'


def extract_qat_inner_weights(keras_path: str) -> Dict[str, List[np.ndarray]]:
    """Return {inner_layer_name: [weight_arrays...]} for every weighted inner layer."""
    with zipfile.ZipFile(keras_path) as z:
        cfg = json.loads(z.read('config.json'))
        weights_blob = z.read('model.weights.h5')

    hf = h5py.File(io.BytesIO(weights_blob), 'r')
    h5_layers = hf['layers']

    wrappers = [L for L in cfg['config']['layers'] if L['class_name'] == 'QuantizeWrapperV2']

    inner_weights: Dict[str, List[np.ndarray]] = {}

    for idx, wrapper in enumerate(wrappers):
        inner = wrapper['config']['layer']
        inner_class = inner['class_name']
        inner_name = inner['config']['name']
        inner_cfg = inner['config']

        h5_name = _wrapper_h5_name(idx)
        if h5_name not in h5_layers:
            raise RuntimeError(f'Missing h5 group {h5_name} for inner layer {inner_name}')
        g = h5_layers[h5_name]

        if inner_class in ('Conv2D', 'DepthwiseConv2D'):
            kernel = g['vars/0'][...]
            weights = [kernel]
            if inner_cfg.get('use_bias', True):
                bias = g['vars/1'][...]
                weights.append(bias)
            inner_weights[inner_name] = weights

        elif inner_class == 'Dense':
            kernel = g['vars/0'][...]
            weights = [kernel]
            if inner_cfg.get('use_bias', True):
                bias = g['vars/1'][...]
                weights.append(bias)
            inner_weights[inner_name] = weights

        elif inner_class == 'BatchNormalization':
            gamma = g['layer/vars/0'][...]
            beta = g['layer/vars/1'][...]
            moving_mean = g['layer/vars/2'][...]
            moving_var = g['layer/vars/3'][...]
            inner_weights[inner_name] = [gamma, beta, moving_mean, moving_var]

        # Pool/Activation/Add/UpSampling/ReLU have no weights -> skip silently.

    hf.close()
    return inner_weights


def transfer_weights(qat_inner_weights, new_model):
    transferred: List[str] = []
    skipped_expected: List[str] = []
    missing: List[str] = []
    mismatched: List[Tuple[str, List, List]] = []

    for layer in new_model.layers:
        if not layer.weights:
            continue

        if layer.name.endswith('_freq_avg_broadcast'):
            skipped_expected.append(layer.name)
            continue

        if layer.name not in qat_inner_weights:
            missing.append(layer.name)
            continue

        src = qat_inner_weights[layer.name]
        dst_shapes = [tuple(w.shape) for w in layer.weights]
        src_shapes = [tuple(w.shape) for w in src]
        if dst_shapes != src_shapes:
            mismatched.append((layer.name, src_shapes, dst_shapes))
            continue

        layer.set_weights(src)
        transferred.append(layer.name)

    return {
        'transferred': transferred,
        'skipped_expected': skipped_expected,
        'missing': missing,
        'mismatched': mismatched,
    }


def convert_to_int8(model, tflite_path):
    ensure_directory(os.path.dirname(tflite_path))
    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i + 1].astype(np.float32)]

    # Export to SavedModel first — TFLite's from_keras_model path on Keras-3
    # functional models hits MLIR "missing attribute 'value'" errors when
    # variables haven't been concretized through a serving signature.
    import tempfile, shutil
    saved_model_dir = tempfile.mkdtemp(prefix='bcresnet_savedmodel_')
    try:
        model.export(saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_quantizer = True
        converter._experimental_disable_per_channel = True

        tflite_bytes = converter.convert()
    finally:
        shutil.rmtree(saved_model_dir, ignore_errors=True)

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
        zero_division=0,
    )
    return acc, macro_f1, report, in_scale, in_zp


def main():
    qat_inner_weights = extract_qat_inner_weights(QAT_CHECKPOINT_PATH)

    new_model = build_bc_resnet(input_shape=(40, 32, 1), num_classes=12)

    result = transfer_weights(qat_inner_weights, new_model)
    if result['missing'] or result['mismatched']:
        msg_parts = ['Weight transfer incomplete.']
        if result['missing']:
            msg_parts.append('Missing in QAT checkpoint: ' + ', '.join(result['missing']))
        if result['mismatched']:
            msg_parts.append('Shape mismatches: ' + str(result['mismatched']))
        raise RuntimeError(' '.join(msg_parts))

    X_test, y_test, _, _ = load_split_dataset('test')
    X_test = X_test[..., np.newaxis].astype(np.float32)
    le = LabelEncoder().fit(KEYWORDS)
    y_test_enc = le.transform(y_test)

    convert_to_int8(new_model, INT8_TFLITE_PATH)

    acc, macro_f1, report, in_scale, in_zp = evaluate_int8(
        INT8_TFLITE_PATH, X_test, y_test_enc, le.classes_,
    )

    int8_size_kb = os.path.getsize(INT8_TFLITE_PATH) / 1024.0

    header = (
        'BC-ResNet INT8 (recovered via h5 weight transfer)\n'
        f'Checkpoint source:  {os.path.relpath(QAT_CHECKPOINT_PATH, PROJECT_ROOT)}\n'
        f'TFLite output:      {os.path.relpath(INT8_TFLITE_PATH, PROJECT_ROOT)}\n'
        f'Test samples:       {len(X_test)} (voicer28-30 originals only)\n'
        f'Transferred layers: {len(result["transferred"])}\n'
        f'Skipped (frozen broadcast): {len(result["skipped_expected"])}\n'
        f'INT8 size:          {int8_size_kb:.2f} KB\n'
        f'INT8 input qparams: scale={in_scale:.6f} zero_point={in_zp}\n'
        f'INT8 test accuracy: {acc * 100:.2f}%\n'
        f'INT8 macro F1:      {macro_f1:.4f}\n'
        '\n'
    )

    ensure_directory(METRICS_DIR)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(report)
        f.write('\n')

    print(header + report)


if __name__ == '__main__':
    main()
