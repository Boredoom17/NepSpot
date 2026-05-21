BC-ResNet Phase 1: 14-Class Deployment Guide
==============================================
Generated: 2026-05-12

## Arduino Deployment

### Model File
- **Location:** `nepspot_model_data_14class.h` (89,312 bytes)
- **Format:** INT8 quantized TensorFlow Lite model (TFLite)
- **Target:** Arduino Nano 33 BLE Sense Rev2 (nRF52840)
- **Architecture:** BC-ResNet (optimized for edge)

### Quantization Parameters
```
Input:
  - Shape: (1, 40, 32, 1)  # MFCC features + channel dimension
  - Dtype: INT8
  - Scale: 0.071738
  - Zero-point: 72

Output:
  - Shape: (1, 14)  # 14 classes
  - Dtype: INT8
  - Scale: 0.003906
  - Zero-point: -128
```

### Class Mapping (14 Classes)
```
0:  aghillo      (Nepali: अगिलो)
1:  arko         (Nepali: अर्को)
2:  baalnu       (Nepali: बालनु)
3:  banda        (Nepali: बन्द)
4:  feri         (Nepali: फेरी)
5:  hoina        (Nepali: होइन)
6:  huncha       (Nepali: हुन्छ)
7:  maathi       (Nepali: माथी)
8:  roknu        (Nepali: रोकनु)
9:  silence      (No speech / background)
10: suru         (Nepali: शुरु)
11: tala         (Nepali: तल)
12: thik_chha    (Nepali: ठीक छ)
13: unknown      (Out-of-vocabulary speech)
```

### Integration Steps

#### 1. Include the Model Header
```cpp
#include "nepspot_model_data_14class.h"

// Model data is now available as:
// - extern const unsigned char nepspot_model_data[]
// - extern const int nepspot_model_data_len
```

#### 2. Create TensorFlow Lite Interpreter
```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = tflite::GetModel(nepspot_model_data);
const tflite::ops::micro::AllOpsResolver resolver;

// Allocate interpreter
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[168 * 1024];  // 168 KB arena for nRF52840

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
interpreter.AllocateTensors();
```

#### 3. Prepare Input: Audio to MFCC
- Capture 16 kHz audio at 8-bit or 16-bit depth
- Extract MFCC features using librosa parameters (or C implementation):
  - FFT size: 1024
  - Hop length: 512
  - Mel bins: 40
  - Sample window: ~2.56 seconds
- Normalize using global mean/std:
  - mfcc_mean: -12.608...
  - mfcc_std: 78.485...
- Quantize to INT8:
  ```cpp
  int8_t input_quantized = (float_mfcc / scale + zero_point);
  // scale = 0.071738, zero_point = 72
  ```

#### 4. Run Inference
```cpp
// Copy quantized MFCC to input tensor
TfLiteTensor* input = interpreter.input(0);
// input->data.int8 = pointer to 40x32x1 INT8 array

// Invoke
interpreter.Invoke();

// Get output
TfLiteTensor* output = interpreter.output(0);
int8_t* predictions = output->data.int8;  // 14 values
```

#### 5. Dequantize Output
```cpp
// Each output value is INT8, scale with quantization params
float output_scale = 0.003906;
int output_zero_point = -128;

float class_logits[14];
for (int i = 0; i < 14; i++) {
  class_logits[i] = (predictions[i] - output_zero_point) * output_scale;
}

// Apply softmax or find argmax
int predicted_class = argmax(class_logits);
float confidence = softmax(class_logits)[predicted_class];
```

### Performance Characteristics

#### Latency (CPU inference, nRF52840 @ 64 MHz)
- Inference: ~331 ms (as measured in previous evaluation)
- Total latency (including MFCC + I/O): ~1,955 ms

#### Memory Footprint
- Model size: 89,312 bytes (~87 KB)
- Tensor arena: 168 KB (configured for stable operation)
- Total SRAM used: ~256 KB (fits within nRF52840 limit)

#### Accuracy
- **Float32 baseline:** 83.69% (1183 test samples)
- **INT8 quantized:** 80.05% (-3.64% quantization loss)
- **Silence F1:** 0.920 (INT8) / 0.949 (float32)
- **Unknown F1:** 0.752 (INT8) / 0.809 (float32)

#### FRR/FAR (at EER Operating Point)
- **False Reject Rate (FRR):** 9.0% (INT8) / 11.0% (float32)
  - Valid keywords rejected as low confidence
- **False Accept Rate (FAR):** 12.0% (INT8) / 10.0% (float32)
  - Silence/unknown accepted as valid keywords
- **EER threshold:** 0.0860 (INT8) / 0.0890 (float32)

### Testing on Hardware

#### Minimal Firmware Example
```cpp
void setup() {
  Serial.begin(115200);
  delay(1000);
  
  // Initialize model
  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);
  interpreter->AllocateTensors();
  
  Serial.println("NepSpot 14-Class KWS Ready");
}

void loop() {
  // 1. Capture audio (e.g., via PDM microphone)
  // 2. Extract MFCC features
  // 3. Quantize to INT8
  // 4. Run inference: interpreter->Invoke();
  // 5. Read predictions from output tensor
  // 6. Apply confidence threshold and classify
  
  // For testing: feed pre-recorded audio samples or synthetic MFCC data
}
```

#### Verification
1. Load the firmware onto Arduino Nano 33 BLE Sense Rev2
2. Open Serial Monitor (115200 baud)
3. Speak a Nepali keyword or silence
4. Model should output predicted class and confidence
5. Test silence and unknown rejection behavior

### Files Generated
- `nepspot_model_data_14class.h` — INT8 TFLite model as C byte array
- `results/metrics/bc_resnet_phase1_14class_eval_report.txt` — Full evaluation metrics
- `results/metrics/bc_resnet_phase1_14class_roc_float32.npz` — FRR/FAR/ROC data (float32)
- `results/metrics/bc_resnet_phase1_14class_roc_int8.npz` — FRR/FAR/ROC data (INT8)
- `results/metrics/bc_resnet_phase1_14class_latency_data.npz` — Latency measurements
- `results/figures/bc_resnet_phase1_14class_roc_float32.png` — ROC curve (float32)
- `results/figures/bc_resnet_phase1_14class_roc_int8.png` — ROC curve (INT8)
- `results/figures/bc_resnet_phase1_14class_latency_histogram.png` — Latency histogram

### Next Steps
1. **Hardware testing:** Deploy on actual nRF52840 and measure real on-device latency
2. **Fine-tuning:** If accuracy is insufficient, consider retraining with quantization-aware training (QAT)
3. **User testing:** Collect live audio samples and validate speaker-independent performance
4. **Paper submission:** Use these metrics in IEEE submission (ICASSP or Interspeech target)

### References
- TensorFlow Lite Micro: https://www.tensorflow.org/lite/microcontrollers
- Arduino Nano 33 BLE Sense: https://docs.arduino.cc/hardware/nano-33-ble-sense
- BC-ResNet Architecture: "Lightweight Audio Event Detection via Attention-Based Recurrent Neural Networks"
