import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

MODELS_DIR    = "models/saved"
TFLITE_DIR    = "models/tflite"
PROCESSED_DIR = "data/processed"
KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

def load_samples(max_samples=200):
    samples, labels = [], []
    for speaker in sorted(os.listdir(PROCESSED_DIR)):
        for wi, word in enumerate(KEYWORDS):
            folder = os.path.join(PROCESSED_DIR, speaker, word)
            if not os.path.exists(folder):
                continue
            files = [f for f in os.listdir(folder) if f.endswith('.npy')]
            for f in files[:2]:
                mfcc = np.load(os.path.join(folder, f))
                samples.append(mfcc)
                labels.append(wi)
                if len(samples) >= max_samples:
                    return np.array(samples, dtype=np.float32)[..., np.newaxis], labels
    return np.array(samples, dtype=np.float32)[..., np.newaxis], labels

def convert():
    os.makedirs(TFLITE_DIR, exist_ok=True)

    saved_model_path = os.path.join(MODELS_DIR, 'saved_model')

    if not os.path.exists(saved_model_path):
        print("ERROR: saved_model folder not found at " + saved_model_path)
        print("Make sure you added model.export() to train.py and retrained.")
        return

    # ── Float32 conversion ──
    print("Converting to TFLite float32...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_float = converter.convert()
    float_path = os.path.join(TFLITE_DIR, 'nepspot_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print("Saved: " + float_path + " (" + str(round(len(tflite_float)/1024, 1)) + " KB)")

    # ── Load representative data ──
    print("\nLoading representative dataset...")
    samples, labels = load_samples(200)
    print("Samples loaded: " + str(len(samples)))

    def representative_dataset():
        for i in range(len(samples)):
            yield [samples[i:i+1]]

    # ── INT8 quantized conversion ──
    print("\nConverting to TFLite INT8...")
    converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_dataset
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter2.inference_input_type  = tf.int8
    converter2.inference_output_type = tf.int8
    tflite_quant = converter2.convert()

    quant_path = os.path.join(TFLITE_DIR, 'nepspot_int8.tflite')
    with open(quant_path, 'wb') as f:
        f.write(tflite_quant)
    print("Saved: " + quant_path + " (" + str(round(len(tflite_quant)/1024, 1)) + " KB)")

    # ── Spot check INT8 accuracy ──
    print("\nVerifying INT8 model on 50 samples...")
    interpreter = tf.lite.Interpreter(model_path=quant_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']

    correct = 0
    for i in range(min(50, len(samples))):
        inp_q = (samples[i:i+1] / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], inp_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == labels[i]:
            correct += 1

    print("INT8 spot-check: " + str(correct) + "/50 = " + str(round(correct/50*100, 1)) + "%")
    print("\nAll done! Files saved to: " + TFLITE_DIR)

if __name__ == "__main__":
    convert()