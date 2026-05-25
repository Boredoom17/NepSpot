import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from data.dataset import KEYWORDS, load_split_dataset

import tensorflow as tf

TFLITE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'tflite', 'nepspot_int8.tflite')


def main():
    X_test, y_test, _, test_speakers = load_split_dataset('test')
    X_test = X_test[..., np.newaxis].astype(np.float32)

    sorted_classes = sorted(KEYWORDS)
    label_to_index = {word: i for i, word in enumerate(sorted_classes)}
    y_true = np.array([label_to_index[w] for w in y_test])

    print(f"Test speakers: {test_speakers}")
    print(f"Test samples: {len(X_test)}")

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    print(f"Input scale={input_scale}, zero_point={input_zero_point}")

    preds = []
    for i in range(len(X_test)):
        x = X_test[i:i+1]
        x_q = (x / input_scale + input_zero_point).round().clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], x_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        preds.append(int(np.argmax(out)))
    preds = np.array(preds)

    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average='macro')

    print()
    print("=== INT8 TFLite — Held-out Speaker Test (voicer28/29/30) ===")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(y_true, preds, target_names=sorted_classes, digits=4))

    print(f"Float32 Keras reference: 84.86% accuracy, macro F1 0.85")
    print(f"Delta (INT8 - float32): {(acc*100 - 84.86):+.2f} pp accuracy, "
          f"{(macro_f1 - 0.85):+.4f} macro F1")


if __name__ == "__main__":
    main()
