import os
import random
import shutil
import sys
import time

SEED = int(os.environ.get('SEED', '42'))

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

random.seed(SEED)

import numpy as np

np.random.seed(SEED)

import tensorflow as tf

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bc_resnet import build_bc_resnet
from phase1_training_utils import build_phase1_train_datasets, inspect_augmentation_behavior
from utils.dataset import (
    KEYWORDS,
    MODELS_DIR,
    RESULTS_DIR,
    ensure_directory,
    load_split_dataset,
    representative_sample_generator,
    summarize_split,
)


BC_BEST_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_best.keras')
BC_FLOAT_BEST_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_float_best.keras')
BC_LABELS_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_label_classes.npy')
BC_SAVED_MODEL_PATH = os.path.join(MODELS_DIR, 'bc_resnet_phase1_saved_model')
BC_TFLITE_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'bc_resnet_int8_phase1.tflite')
BC_HEADER_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'bc_resnet_int8_phase1.h')
BC_REPORT_PATH = os.path.join(RESULTS_DIR, 'metrics', 'bc_resnet_phase1_report.txt')

DS_BEST_PATH = os.path.join(MODELS_DIR, 'best_model.keras')
DS_TFLITE_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'nepspot_int8.tflite')
VANILLA_BEST_PATH = os.path.join(MODELS_DIR, 'vanilla_cnn_best.keras')
VANILLA_TFLITE_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'vanilla_cnn_int8.tflite')


def count_parameters(model):
    trainable_params = int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))
    non_trainable_params = int(np.sum([np.prod(weight.shape) for weight in model.non_trainable_weights]))
    total_params = trainable_params + non_trainable_params
    return total_params, trainable_params, non_trainable_params


def model_size_kb(total_params):
    return (total_params * 4) / 1024.0


def prepare_dataset():
    train_summary = summarize_split('train')
    val_summary = summarize_split('val')
    test_summary = summarize_split('test')

    print('Training speakers: ' + str(train_summary['num_speakers']))
    print('Validation speakers: ' + str(val_summary['num_speakers']))
    print('Test speakers: ' + str(test_summary['num_speakers']))

    X_train, y_train, _, train_speakers = load_split_dataset('train')
    X_val, y_val, _, val_speakers = load_split_dataset('val')
    X_test, y_test, _, test_speakers = load_split_dataset('test')

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)
    X_test = X_test[..., np.newaxis].astype(np.float32)

    le = LabelEncoder()
    le.fit(KEYWORDS)
    assert len(le.classes_) == 12, f'Expected 12 classes, got {len(le.classes_)}: {list(le.classes_)}'
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    y_train_onehot = tf.one_hot(y_train_encoded, depth=len(KEYWORDS))
    y_val_onehot = tf.one_hot(y_val_encoded, depth=len(KEYWORDS))
    y_test_onehot = tf.one_hot(y_test_encoded, depth=len(KEYWORDS))

    print('Classes: ' + str(list(le.classes_)))
    print('Train speakers: ' + ', '.join(train_speakers))
    print('Val speakers: ' + ', '.join(val_speakers))
    print('Test speakers: ' + ', '.join(test_speakers))
    print('Train: ' + str(len(X_train)) + ' samples')
    print('Val: ' + str(len(X_val)) + ' samples')
    print('Test: ' + str(len(X_test)) + ' samples')

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot, y_test_encoded, le, test_speakers, test_summary


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.1,
        ),
        metrics=['accuracy'],
    )


def train_model(model, X_train, y_train, X_val, y_val):
    ensure_directory(MODELS_DIR)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BC_FLOAT_BEST_PATH,
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

    train_ds, debug_ds = build_phase1_train_datasets(X_train, y_train, batch_size=32, shuffle_buffer=4096)
    aug_stats = inspect_augmentation_behavior(debug_ds, max_batches=12)
    print(
        'Phase1 augment probe: specaug_examples=' + str(aug_stats['specaug_examples']) +
        ', zero_mask_examples=' + str(aug_stats['zero_mask_examples']) +
        ', mixup_batches=' + str(aug_stats['mixup_batches']) +
        ', observed_batches=' + str(aug_stats['observed_batches'])
    )

    print('\nStarting training...')
    model.fit(
        train_ds,
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=callbacks,
        verbose=1,
    )


def build_bc_resnet_tfkeras(input_shape=(40, 32, 1), num_classes=12):
    """tf_keras (Keras-2) port of bc_resnet.build_bc_resnet for tfmot QAT.

    Mirrors the Kim 2021 BC-ResNet-1 architecture in src/models/bc_resnet.py:
      - frequency branch: DepthwiseConv2D(kernel=(3,1), no dilation) -> BN
      - time branch:      DepthwiseConv2D(kernel=(1,3), dilation=(1,d)) -> BN -> Swish
      - broadcast: AveragePooling2D over freq -> UpSampling2D(nearest) back to F
      - branch_add -> Conv2D(1x1) -> BN -> (+residual if shapes match) -> ReLU

    UpSampling2D is used here in place of the bc_resnet.py Lambda(tile) so the
    resulting graph is fully composed of standard Keras layers that tfmot can
    wrap. Weight layer order and shapes match bc_resnet.py exactly, so
    tfk_model.set_weights(float_model.get_weights()) transfers cleanly.
    """
    import tf_keras as keras

    layers = keras.layers
    models = keras.models

    def _freq_branch(x, name):
        y = layers.DepthwiseConv2D(
            kernel_size=(3, 1),
            padding='same',
            use_bias=False,
            name=f'{name}_freq_dw',
        )(x)
        y = layers.BatchNormalization(name=f'{name}_freq_bn')(y)
        return y

    def _time_branch(x, time_dilation, name):
        y = layers.DepthwiseConv2D(
            kernel_size=(1, 3),
            dilation_rate=(1, time_dilation),
            padding='same',
            use_bias=False,
            name=f'{name}_time_dw',
        )(x)
        y = layers.BatchNormalization(name=f'{name}_time_bn')(y)
        y = layers.Activation('swish', name=f'{name}_time_swish')(y)
        return y

    def _broadcast_over_freq(t, name):
        # Frozen DepthwiseConv2D with kernel (2F-1, 1) and weights = 1/F.
        # Single op equivalent of AvgPool((F,1)) + UpSampling2D((F,1)), but
        # quantizes cleanly through tfmot QAT + TFLite INT8 conversion (the
        # AvgPool+UpSample variant triggers a 'same scale constraint' error).
        freq_dim = t.shape[1]
        if freq_dim is None:
            raise ValueError('BC-ResNet expects a static frequency dimension')
        freq_dim = int(freq_dim)
        kernel_h = 2 * freq_dim - 1

        broadcaster = layers.DepthwiseConv2D(
            kernel_size=(kernel_h, 1),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            depthwise_initializer=keras.initializers.Constant(1.0 / freq_dim),
            trainable=False,
            name=f'{name}_freq_avg_broadcast',
        )
        return broadcaster(t)

    def _normal_block(x, filters, time_dilation, name):
        shortcut = x

        f = _freq_branch(x, name=f'{name}_fb')
        t = _time_branch(x, time_dilation=time_dilation, name=f'{name}_tb')
        t_bcast = _broadcast_over_freq(t, name=f'{name}_tb_bcast')

        y = layers.Add(name=f'{name}_branch_add')([f, t_bcast])

        y = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            name=f'{name}_pw',
        )(y)
        y = layers.BatchNormalization(name=f'{name}_pw_bn')(y)

        if (shortcut.shape[-1] == filters
                and shortcut.shape[1] == y.shape[1]
                and shortcut.shape[2] == y.shape[2]):
            y = layers.Add(name=f'{name}_res_add')([y, shortcut])

        y = layers.ReLU(name=f'{name}_relu')(y)
        return y

    def _transition_block(x, filters, name):
        x = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            name=f'{name}_pw',
        )(x)
        x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
        x = layers.ReLU(name=f'{name}_relu')(x)
        x = layers.AveragePooling2D(
            pool_size=(2, 1),
            strides=(2, 1),
            padding='same',
            name=f'{name}_freq_pool',
        )(x)
        return x

    inputs = keras.Input(shape=input_shape, name='mfcc_input')

    x = layers.Conv2D(
        16,
        kernel_size=(5, 5),
        strides=(2, 1),
        padding='same',
        use_bias=False,
        name='stem_conv',
    )(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)

    x = _transition_block(x, filters=8, name='stage1_transition')
    x = _normal_block(x, filters=8,  time_dilation=1, name='stage1_block1')
    x = _normal_block(x, filters=8,  time_dilation=1, name='stage1_block2')

    x = _transition_block(x, filters=12, name='stage2_transition')
    x = _normal_block(x, filters=12, time_dilation=2, name='stage2_block1')
    x = _normal_block(x, filters=12, time_dilation=2, name='stage2_block2')

    x = _transition_block(x, filters=16, name='stage3_transition')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block1')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block2')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block3')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block4')

    x = _transition_block(x, filters=20, name='stage4_transition')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block1')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block2')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block3')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block4')

    x = layers.Conv2D(
        32, kernel_size=(1, 5), padding='same', use_bias=False, name='head_conv1',
    )(x)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.ReLU(name='head_relu1')(x)
    x = layers.Conv2D(
        32, kernel_size=(1, 1), padding='same', use_bias=False, name='head_conv2',
    )(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.ReLU(name='head_relu2')(x)

    x = layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    return models.Model(inputs, outputs, name='BC_ResNet_NepSpot_TFKeras')


def run_qat_finetuning(model, X_train, y_train, X_val, y_val):
    try:
        import tensorflow_model_optimization as tfmot
    except Exception as exc:
        raise RuntimeError(
            'QAT requires tensorflow-model-optimization. Install with: '
            '/Users/ad/codes/.venv310/bin/python3 -m pip install tensorflow-model-optimization'
        ) from exc

    try:
        qat_model = tfmot.quantization.keras.quantize_model(model)
    except Exception:
        try:
            tfk_model = build_bc_resnet_tfkeras(input_shape=(40, 32, 1), num_classes=len(KEYWORDS))
            tfk_model.set_weights(model.get_weights())
            qat_model = tfmot.quantization.keras.quantize_model(tfk_model)
        except Exception as exc:
            layer_summary = ', '.join([f'{layer.name}:{layer.__class__.__name__}' for layer in model.layers])
            raise RuntimeError(
                'QAT wrapping failed. Potential incompatible layers: ' + layer_summary +
                '. Alternatives: manual quantize_annotate_layer wrapping or PTQ fallback.'
            ) from exc

    if qat_model.__class__.__module__.startswith('tf_keras'):
        import tf_keras as keras_backend
    else:
        keras_backend = tf.keras

    qat_model.compile(
        optimizer=keras_backend.optimizers.Adam(learning_rate=1e-4),
        loss=keras_backend.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.1,
        ),
        metrics=['accuracy'],
    )

    callbacks = [
        keras_backend.callbacks.ModelCheckpoint(
            filepath=BC_BEST_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        keras_backend.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_backend.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    train_ds, debug_ds = build_phase1_train_datasets(X_train, y_train, batch_size=32, shuffle_buffer=4096)
    aug_stats = inspect_augmentation_behavior(debug_ds, max_batches=12)
    print(
        'QAT augment probe: specaug_examples=' + str(aug_stats['specaug_examples']) +
        ', zero_mask_examples=' + str(aug_stats['zero_mask_examples']) +
        ', mixup_batches=' + str(aug_stats['mixup_batches']) +
        ', observed_batches=' + str(aug_stats['observed_batches'])
    )

    print('\nStarting QAT fine-tuning...')
    qat_model.fit(
        train_ds,
        validation_data=(X_val, y_val),
        epochs=15,
        callbacks=callbacks,
        verbose=1,
    )
    return qat_model


def evaluate_keras_model(model, X_test, y_test, label_names):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    if len(np.shape(y_test)) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
    )
    return loss, accuracy, macro_f1, y_pred, report


def measure_cpu_inference_time(model, X_test, sample_count=100):
    total_samples = min(sample_count, len(X_test))
    if total_samples == 0:
        return 0.0

    samples = X_test[:total_samples]

    for sample in samples[:5]:
        with tf.device('/CPU:0'):
            _ = model(sample[np.newaxis, ...], training=False)

    start = time.perf_counter()
    for sample in samples:
        with tf.device('/CPU:0'):
            _ = model(sample[np.newaxis, ...], training=False)
    elapsed = time.perf_counter() - start
    return (elapsed / total_samples) * 1000.0


def save_report(
    report_path,
    test_summary,
    float32_metrics,
    float32_pre_qat_metrics,
    int8_metrics,
    total_params,
    trainable_params,
    non_trainable_params,
):
    float32_loss, float32_accuracy, float32_macro_f1, float32_inference_ms, classification_text = float32_metrics
    float32_pre_qat_accuracy, float32_pre_qat_macro_f1 = float32_pre_qat_metrics
    int8_accuracy, int8_macro_f1, int8_size_kb = int8_metrics

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('Evaluation split: speaker-independent test\n')
        handle.write('Speakers: ' + ', '.join(test_summary['speakers']) + '\n')
        handle.write('Samples: ' + str(test_summary['num_samples']) + '\n\n')
        handle.write('Model: BC-ResNet-1 baseline\n')
        handle.write('Reference: Qualcomm AI Research BC-ResNet Keras port\n')
        handle.write('Float32 pre-QAT test accuracy: ' + f'{float32_pre_qat_accuracy * 100:.2f}%' + '\n')
        handle.write('Float32 pre-QAT macro F1: ' + f'{float32_pre_qat_macro_f1:.4f}' + '\n')
        handle.write('Total parameters: ' + str(total_params) + '\n')
        handle.write('Trainable parameters: ' + str(trainable_params) + '\n')
        handle.write('Non-trainable parameters: ' + str(non_trainable_params) + '\n')
        handle.write('Float32 model size (KB): ' + f'{model_size_kb(total_params):.2f}' + '\n')
        handle.write('Float32 test accuracy: ' + f'{float32_accuracy * 100:.2f}%' + '\n')
        handle.write('Float32 macro F1: ' + f'{float32_macro_f1:.4f}' + '\n')
        handle.write('CPU inference time per sample (ms): ' + f'{float32_inference_ms:.3f}' + '\n')
        handle.write('INT8 model size (KB): ' + f'{int8_size_kb:.2f}' + '\n')
        handle.write('INT8 test accuracy: ' + f'{int8_accuracy * 100:.2f}%' + '\n')
        handle.write('INT8 macro F1: ' + f'{int8_macro_f1:.4f}' + '\n\n')
        handle.write(classification_text)


def quantize_to_int8(saved_model_path, tflite_path):
    ensure_directory(os.path.dirname(tflite_path))

    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    if not os.path.exists(saved_model_path):
        raise FileNotFoundError('SavedModel folder not found at ' + saved_model_path)

    samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for index in range(len(samples)):
            yield [samples[index:index + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as handle:
        handle.write(tflite_model)

    return tflite_model


def quantize_qat_model_to_int8(qat_model, tflite_path):
    ensure_directory(os.path.dirname(tflite_path))

    if os.path.exists(tflite_path):
        os.remove(tflite_path)

    samples, _ = representative_sample_generator(split_name='train', max_samples=200)
    if len(samples) == 0:
        raise RuntimeError('No representative samples available for INT8 conversion')

    def representative_dataset():
        for index in range(len(samples)):
            yield [samples[index:index + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # BC-ResNet's broadcast path (AvgPool over freq -> UpSampling2D -> Add)
    # makes the AvgPool output share a tensor with a different-scaled neighbor
    # through the Add, which trips the converter's "same scale constraint".
    # The three flags below collectively let TFLite reconcile that:
    #   - experimental_enable_resource_variables: required by the converter
    #     error message itself (variable constant folding).
    #   - experimental_new_quantizer: MLIR-based quantizer handles
    #     same-scale ops (AvgPool, Concat, Reshape) more flexibly than legacy.
    #   - _experimental_disable_per_channel: per-tensor (not per-channel)
    #     quantization avoids scale conflicts at the Add fan-in.
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_quantizer = True
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as handle:
        handle.write(tflite_model)

    return tflite_model


def evaluate_int8_tflite(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    if input_scale == 0:
        raise RuntimeError('Invalid INT8 quantization scale for input tensor')

    predictions = []
    for sample in X_test:
        quantized_sample = np.round(sample / input_scale + input_zero_point).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], quantized_sample[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(int(np.argmax(output)))

    predictions = np.array(predictions)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average='macro')
    return accuracy, macro_f1


def write_tflite_header(tflite_path, header_path, array_name='nepspot_model_data'):
    with open(tflite_path, 'rb') as handle:
        model_bytes = handle.read()

    include_guard = f'{array_name.upper()}_H_'
    hex_bytes = [f'0x{byte:02x}' for byte in model_bytes]
    lines = []
    for index in range(0, len(hex_bytes), 12):
        lines.append('  ' + ', '.join(hex_bytes[index:index + 12]))
    body = ',\n'.join(lines)

    content = (
        f'#ifndef {include_guard}\n'
        f'#define {include_guard}\n\n'
        '#include <stdint.h>\n\n'
        f'const unsigned char {array_name}[] = {{\n'
        f'{body}\n'
        '};\n\n'
        f'const unsigned int {array_name}_len = {len(model_bytes)};\n\n'
        f'#endif  // {include_guard}\n'
    )

    with open(header_path, 'w', encoding='utf-8') as handle:
        handle.write(content)


def load_reference_model_metrics(name, keras_path, tflite_path, X_test, y_test):
    if not os.path.exists(keras_path):
        return None

    try:
        model = tf.keras.models.load_model(keras_path)
    except Exception as exc:
        print(f'Could not load {name} model from {keras_path}: {exc}')
        return None

    total_params, trainable_params, non_trainable_params = count_parameters(model)
    _, float32_accuracy, float32_macro_f1, _, _ = evaluate_keras_model(model, X_test, y_test, sorted(KEYWORDS))
    float32_inference_ms = measure_cpu_inference_time(model, X_test)

    int8_accuracy = None
    int8_macro_f1 = None
    int8_size_kb = None
    if os.path.exists(tflite_path):
        int8_accuracy, int8_macro_f1 = evaluate_int8_tflite(tflite_path, X_test, y_test)
        int8_size_kb = os.path.getsize(tflite_path) / 1024.0

    return {
        'name': name,
        'total_params': total_params,
        'float32_size_kb': model_size_kb(total_params),
        'int8_size_kb': int8_size_kb,
        'float32_accuracy': float32_accuracy,
        'int8_accuracy': int8_accuracy,
        'macro_f1': float32_macro_f1,
        'cpu_inference_ms': float32_inference_ms,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
    }


def placeholder_metrics(name):
    return {
        'name': name,
        'total_params': None,
        'float32_size_kb': None,
        'int8_size_kb': None,
        'float32_accuracy': None,
        'int8_accuracy': None,
        'macro_f1': None,
        'cpu_inference_ms': None,
    }


def print_comparison_table(rows):
    headers = ['Model', 'Params', 'Float32 KB', 'INT8 KB', 'Float32 Acc', 'INT8 Acc', 'Macro F1', 'CPU ms/sample']

    def format_value(value, precision=2):
        if value is None:
            return 'n/a'
        if isinstance(value, int):
            return str(value)
        return f'{value:.{precision}f}'

    table_rows = []
    for row in rows:
        table_rows.append([
            row['name'],
            format_value(row['total_params'], 0),
            format_value(row['float32_size_kb'], 2),
            format_value(row.get('int8_size_kb'), 2),
            format_value(row['float32_accuracy'] * 100, 2),
            format_value(row.get('int8_accuracy') * 100 if row.get('int8_accuracy') is not None else None, 2),
            format_value(row['macro_f1'], 4),
            format_value(row['cpu_inference_ms'], 3),
        ])

    widths = [len(header) for header in headers]
    for row in table_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def render_row(values):
        return ' | '.join(str(value).ljust(widths[index]) for index, value in enumerate(values))

    separator = '-+-'.join('-' * width for width in widths)
    print('\nComparison summary:')
    print(render_row(headers))
    print(separator)
    for row in table_rows:
        print(render_row(row))


def main():
    ensure_directory(RESULTS_DIR)
    ensure_directory(os.path.join(RESULTS_DIR, 'metrics'))
    ensure_directory(os.path.join(os.path.dirname(MODELS_DIR), 'tflite'))

    X_train, y_train, X_val, y_val, X_test, y_test, y_test_encoded, le, test_speakers, test_summary = prepare_dataset()

    model = build_bc_resnet(
        input_shape=(40, 32, 1),
        num_classes=len(KEYWORDS),
    )

    compile_model(model)
    train_model(model, X_train, y_train, X_val, y_val)

    print('\nEvaluating float32 model before QAT on test set...')
    float32_pre_loss, float32_pre_accuracy, float32_pre_macro_f1, _, _ = evaluate_keras_model(
        model,
        X_test,
        y_test,
        le.classes_,
    )
    print('Pre-QAT float32 accuracy: ' + str(round(float32_pre_accuracy * 100, 2)) + '%')
    print('Pre-QAT float32 loss: ' + str(round(float32_pre_loss, 4)))
    print('Pre-QAT float32 macro F1: ' + str(round(float32_pre_macro_f1, 4)))

    qat_model = run_qat_finetuning(model, X_train, y_train, X_val, y_val)

    print('\nEvaluating float32 model after QAT on test set...')
    float32_loss, float32_accuracy, float32_macro_f1, _, classification_text = evaluate_keras_model(
        qat_model,
        X_test,
        y_test,
        le.classes_,
    )
    float32_inference_ms = measure_cpu_inference_time(qat_model, X_test)

    print('Test accuracy: ' + str(round(float32_accuracy * 100, 2)) + '%')
    print('Test loss: ' + str(round(float32_loss, 4)))
    print('Macro F1: ' + str(round(float32_macro_f1, 4)))
    print('CPU inference time/sample: ' + str(round(float32_inference_ms, 3)) + ' ms')

    total_params, trainable_params, non_trainable_params = count_parameters(model)
    np.save(BC_LABELS_PATH, le.classes_)

    if os.path.exists(BC_SAVED_MODEL_PATH):
        shutil.rmtree(BC_SAVED_MODEL_PATH)
    qat_model.export(BC_SAVED_MODEL_PATH)

    print('Saved best Keras model to: ' + BC_BEST_PATH)
    print('Saved float32 pre-QAT checkpoint to: ' + BC_FLOAT_BEST_PATH)
    print('Saved labels to: ' + BC_LABELS_PATH)
    print('SavedModel exported to: ' + BC_SAVED_MODEL_PATH)

    print('\nConverting to INT8 TFLite...')
    tflite_bytes = quantize_qat_model_to_int8(qat_model, BC_TFLITE_PATH)
    int8_size_kb = len(tflite_bytes) / 1024.0
    int8_accuracy, int8_macro_f1 = evaluate_int8_tflite(BC_TFLITE_PATH, X_test, y_test_encoded)

    print('Saved INT8 model to: ' + BC_TFLITE_PATH + ' (' + str(round(int8_size_kb, 2)) + ' KB)')
    print('INT8 test accuracy: ' + str(round(int8_accuracy * 100, 2)) + '%')
    print('INT8 macro F1: ' + str(round(int8_macro_f1, 4)))

    write_tflite_header(BC_TFLITE_PATH, BC_HEADER_PATH, array_name='nepspot_model_data')
    print('Saved Arduino header to: ' + BC_HEADER_PATH)

    test_summary = dict(test_summary)
    test_summary['speakers'] = test_speakers
    save_report(
        BC_REPORT_PATH,
        test_summary,
        (float32_loss, float32_accuracy, float32_macro_f1, float32_inference_ms, classification_text),
        (float32_pre_accuracy, float32_pre_macro_f1),
        (int8_accuracy, int8_macro_f1, int8_size_kb),
        total_params,
        trainable_params,
        non_trainable_params,
    )
    print('Saved report to: ' + BC_REPORT_PATH)

    bc_metrics = {
        'name': 'BC-ResNet',
        'total_params': total_params,
        'float32_size_kb': model_size_kb(total_params),
        'int8_size_kb': int8_size_kb,
        'float32_accuracy': float32_accuracy,
        'int8_accuracy': int8_accuracy,
        'macro_f1': float32_macro_f1,
        'cpu_inference_ms': float32_inference_ms,
    }

    ds_metrics = load_reference_model_metrics('DS-CNN', DS_BEST_PATH, DS_TFLITE_PATH, X_test, y_test_encoded)
    vanilla_metrics = load_reference_model_metrics('Vanilla CNN', VANILLA_BEST_PATH, VANILLA_TFLITE_PATH, X_test, y_test_encoded)

    comparison_rows = [
        vanilla_metrics if vanilla_metrics is not None else placeholder_metrics('Vanilla CNN'),
        ds_metrics if ds_metrics is not None else placeholder_metrics('DS-CNN'),
        bc_metrics,
    ]

    print_comparison_table(comparison_rows)


if __name__ == '__main__':
    main()