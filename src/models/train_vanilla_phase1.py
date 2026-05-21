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

from vanilla_cnn import build_vanilla_cnn
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


VANILLA_BEST_PATH = os.path.join(MODELS_DIR, 'vanilla_cnn_phase1_best.keras')
VANILLA_LABELS_PATH = os.path.join(MODELS_DIR, 'vanilla_cnn_phase1_label_classes.npy')
VANILLA_SAVED_MODEL_PATH = os.path.join(MODELS_DIR, 'vanilla_cnn_phase1_saved_model')
VANILLA_TFLITE_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'vanilla_cnn_int8_phase1.tflite')
VANILLA_REPORT_PATH = os.path.join(RESULTS_DIR, 'metrics', 'vanilla_cnn_phase1_report.txt')

DS_BEST_PATH = os.path.join(MODELS_DIR, 'best_model.keras')
DS_LABELS_PATH = os.path.join(MODELS_DIR, 'label_classes.npy')
DS_TFLITE_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'tflite', 'nepspot_int8.tflite')


def count_parameters(model):
    """Return (total, trainable, non-trainable) parameter counts for `model`."""
    trainable_params = int(np.sum([np.prod(weight.shape) for weight in model.trainable_weights]))
    non_trainable_params = int(np.sum([np.prod(weight.shape) for weight in model.non_trainable_weights]))
    total_params = trainable_params + non_trainable_params
    return total_params, trainable_params, non_trainable_params


def model_size_kb(total_params):
    """Estimate float32 model size in KB from parameter count."""
    return (total_params * 4) / 1024.0


def prepare_dataset():
    """Load and prepare train/val/test arrays and label encoder for Vanilla Phase1."""
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

    return (
        X_train,
        y_train_onehot,
        X_val,
        y_val_onehot,
        X_test,
        y_test_onehot,
        y_test_encoded,
        le,
        test_speakers,
        test_summary,
    )


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.1,
        ),
        metrics=['accuracy']
    )


def train_model(model, X_train, y_train, X_val, y_val):
    ensure_directory(MODELS_DIR)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=VANILLA_BEST_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
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
        verbose=1
    )


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


def save_report(report_path, test_summary, float32_metrics, int8_metrics, total_params, trainable_params, non_trainable_params):
    float32_loss, float32_accuracy, float32_macro_f1, float32_inference_ms, classification_text = float32_metrics
    int8_accuracy, int8_macro_f1, int8_size_kb = int8_metrics

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('Evaluation split: speaker-independent test\n')
        handle.write('Speakers: ' + ', '.join(test_summary['speakers']) + '\n')
        handle.write('Samples: ' + str(test_summary['num_samples']) + '\n\n')
        handle.write('Model: Vanilla CNN baseline\n')
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


def load_reference_dscnn_metrics(X_test, y_test):
    if not os.path.exists(DS_BEST_PATH):
        return None

    model = tf.keras.models.load_model(DS_BEST_PATH)
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    _, float32_accuracy, float32_macro_f1, _, _ = evaluate_keras_model(model, X_test, y_test, sorted(KEYWORDS))
    float32_inference_ms = measure_cpu_inference_time(model, X_test)

    int8_accuracy = None
    int8_macro_f1 = None
    if os.path.exists(DS_TFLITE_PATH):
        int8_accuracy, int8_macro_f1 = evaluate_int8_tflite(DS_TFLITE_PATH, X_test, y_test)

    return {
        'name': 'DS-CNN',
        'total_params': total_params,
        'float32_size_kb': model_size_kb(total_params),
        'int8_size_kb': (os.path.getsize(DS_TFLITE_PATH) / 1024.0) if os.path.exists(DS_TFLITE_PATH) else None,
        'float32_accuracy': float32_accuracy,
        'int8_accuracy': int8_accuracy,
        'macro_f1': float32_macro_f1,
        'cpu_inference_ms': float32_inference_ms,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
    }


def print_comparison_table(ds_metrics, vanilla_metrics):
    rows = [
        ds_metrics,
        vanilla_metrics,
    ]
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

    model = build_vanilla_cnn(
        input_shape=(40, 32, 1),
        num_classes=len(KEYWORDS)
    )

    compile_model(model)

    train_model(model, X_train, y_train, X_val, y_val)

    print('\nEvaluating on test set...')
    float32_loss, float32_accuracy, float32_macro_f1, _, classification_text = evaluate_keras_model(model, X_test, y_test, le.classes_)
    float32_inference_ms = measure_cpu_inference_time(model, X_test)

    print('Test accuracy: ' + str(round(float32_accuracy * 100, 2)) + '%')
    print('Test loss: ' + str(round(float32_loss, 4)))
    print('Macro F1: ' + str(round(float32_macro_f1, 4)))
    print('CPU inference time/sample: ' + str(round(float32_inference_ms, 3)) + ' ms')

    total_params, trainable_params, non_trainable_params = count_parameters(model)
    float32_size = model_size_kb(total_params)

    np.save(VANILLA_LABELS_PATH, le.classes_)

    if os.path.exists(VANILLA_SAVED_MODEL_PATH):
        shutil.rmtree(VANILLA_SAVED_MODEL_PATH)
    model.export(VANILLA_SAVED_MODEL_PATH)

    print('Saved best Keras model to: ' + VANILLA_BEST_PATH)
    print('Saved labels to: ' + VANILLA_LABELS_PATH)
    print('SavedModel exported to: ' + VANILLA_SAVED_MODEL_PATH)

    print('\nConverting to INT8 TFLite...')
    tflite_bytes = quantize_to_int8(VANILLA_SAVED_MODEL_PATH, VANILLA_TFLITE_PATH)
    int8_size_kb = len(tflite_bytes) / 1024.0
    int8_accuracy, int8_macro_f1 = evaluate_int8_tflite(VANILLA_TFLITE_PATH, X_test, y_test_encoded)

    print('Saved INT8 model to: ' + VANILLA_TFLITE_PATH + ' (' + str(round(int8_size_kb, 2)) + ' KB)')
    print('INT8 test accuracy: ' + str(round(int8_accuracy * 100, 2)) + '%')
    print('INT8 macro F1: ' + str(round(int8_macro_f1, 4)))

    test_summary = dict(test_summary)
    test_summary['speakers'] = test_speakers
    save_report(
        VANILLA_REPORT_PATH,
        test_summary,
        (float32_loss, float32_accuracy, float32_macro_f1, float32_inference_ms, classification_text),
        (int8_accuracy, int8_macro_f1, int8_size_kb),
        total_params,
        trainable_params,
        non_trainable_params,
    )
    print('Saved report to: ' + VANILLA_REPORT_PATH)

    vanilla_metrics = {
        'name': 'Vanilla CNN',
        'total_params': total_params,
        'float32_size_kb': float32_size,
        'int8_size_kb': int8_size_kb,
        'float32_accuracy': float32_accuracy,
        'int8_accuracy': int8_accuracy,
        'macro_f1': float32_macro_f1,
        'cpu_inference_ms': float32_inference_ms,
    }

    ds_metrics = load_reference_dscnn_metrics(X_test, y_test_encoded)
    if ds_metrics is not None:
        print_comparison_table(ds_metrics, vanilla_metrics)
    else:
        print('\nDS-CNN reference artifacts not found, skipping comparison table.')


if __name__ == '__main__':
    main()