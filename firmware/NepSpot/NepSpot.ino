#include <Arduino.h>
#include <PDM.h>
#include <math.h>
#include <arm_math.h>

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_int8.h"

namespace {

// ================= CONFIG =================
constexpr int kSampleRate   = 16000;
constexpr int kAudioSamples = 16000;
constexpr int kNfft         = 1024;
constexpr int kHopLength    = 512;
constexpr int kNFreqBins    = (kNfft / 2) + 1;
constexpr int kNMfcc        = 40;
constexpr int kNumFrames    = 32;
constexpr int kKeywordCount = 12;
constexpr int kPad          = kNfft / 2;

constexpr float kMfccMean = -12.417611f;
constexpr float kMfccStd  =  77.941345f;
constexpr float kRmsThreshold = 0.006f;

// ================= AUDIO (int16 — full resolution, matches training) =================
volatile int  g_audioWritePos = 0;
volatile bool g_audioReady    = false;
int16_t g_audioBuffer[kAudioSamples];  // 32KB, full 16-bit PCM

// ================= FFT / MEL / MFCC BUFFERS =================
static float g_fft_input[kNfft];
static float g_fft_output[kNfft];
static arm_rfft_fast_instance_f32 g_fft_instance;
static float g_mel[kNMfcc];
static float g_mfcc[kNMfcc];
int g_melBins[kNMfcc + 2];

// ================= TFLM =================
const tflite::Model*       g_model       = nullptr;
tflite::MicroInterpreter*  g_interpreter = nullptr;
TfLiteTensor*              g_input       = nullptr;
TfLiteTensor*              g_output      = nullptr;

// ================= KEYWORDS (12 classes) =================
const char* kKeywords[kKeywordCount] = {
  "aghillo","arko","baalnu","banda",
  "feri","hoina","huncha","maathi",
  "roknu","suru","tala","thik_chha"
};

// ================= AUDIO CALLBACK =================
void onPDMdata() {
  int bytesAvailable = PDM.available();
  int16_t buffer[256];
  while (bytesAvailable > 0) {
    int bytesRead = PDM.read(buffer, sizeof(buffer));
    int samples   = bytesRead / 2;
    for (int i = 0; i < samples; i++) {
      if (g_audioWritePos < kAudioSamples)
        g_audioBuffer[g_audioWritePos++] = buffer[i];  // full 16-bit, no >> 8
    }
    if (g_audioWritePos >= kAudioSamples) g_audioReady = true;
    bytesAvailable -= bytesRead;
  }
}

void reset_audio() {
  noInterrupts();
  g_audioWritePos = 0;
  g_audioReady    = false;
  interrupts();
}

void countdown() {
  Serial.println("3..."); delay(1000);
  Serial.println("2..."); delay(1000);
  Serial.println("1..."); delay(1000);
  reset_audio();
  Serial.println(">>> SPEAK NOW! <<<");
}

// ================= SLANEY MEL SCALE =================
float hz_to_mel(float hz) {
  const float f_sp        = 200.0f / 3.0f;
  const float min_log_hz  = 1000.0f;
  const float min_log_mel = min_log_hz / f_sp;
  const float logstep     = logf(6.4f) / 27.0f;
  return (hz < min_log_hz)
    ? hz / f_sp
    : min_log_mel + logf(hz / min_log_hz) / logstep;
}

float mel_to_hz(float mel) {
  const float f_sp        = 200.0f / 3.0f;
  const float min_log_hz  = 1000.0f;
  const float min_log_mel = min_log_hz / f_sp;
  const float logstep     = logf(6.4f) / 27.0f;
  return (mel < min_log_mel)
    ? f_sp * mel
    : min_log_hz * expf(logstep * (mel - min_log_mel));
}

void init_mel() {
  float mel_min = hz_to_mel(0.0f);
  float mel_max = hz_to_mel(kSampleRate / 2.0f);
  for (int i = 0; i < kNMfcc + 2; i++) {
    float mel = mel_min + (mel_max - mel_min) * i / (kNMfcc + 1);
    float hz  = mel_to_hz(mel);
    g_melBins[i] = (int)(hz * kNfft / kSampleRate);
    if (g_melBins[i] >= kNFreqBins) g_melBins[i] = kNFreqBins - 1;
  }
}

// ================= FFT =================
void compute_fft(int start) {
  for (int i = 0; i < kNfft; i++) {
    int audio_idx = (start + i) - kPad;
    float s = (audio_idx >= 0 && audio_idx < kAudioSamples)
              ? g_audioBuffer[audio_idx] / 32768.0f : 0.0f;  // full 16-bit scale
    float w = 0.5f * (1.0f - cosf(2.0f * PI * i / (kNfft - 1)));
    g_fft_input[i] = s * w;
  }
  arm_rfft_fast_f32(&g_fft_instance, g_fft_input, g_fft_output, 0);
}

// ================= MEL FILTERBANK =================
void mel_filter() {
  for (int m = 1; m <= kNMfcc; m++) {
    int l = g_melBins[m - 1];
    int c = g_melBins[m];
    int r = g_melBins[m + 1];
    if (c <= l || r <= c) {
      g_mel[m - 1] = 10.0f * log10f(1e-10f);
      continue;
    }
    float e = 0.0f;
    for (int k = l; k < c; k++) {
      float p;
      if      (k == 0)            p = g_fft_output[0] * g_fft_output[0];
      else if (k == kNFreqBins-1) p = g_fft_output[1] * g_fft_output[1];
      else { float re = g_fft_output[2*k], im = g_fft_output[2*k+1]; p = re*re + im*im; }
      float w = (float)(k - l) / (float)(c - l);
      e += w * p;
    }
    for (int k = c; k <= r; k++) {
      float p;
      if      (k == 0)            p = g_fft_output[0] * g_fft_output[0];
      else if (k == kNFreqBins-1) p = g_fft_output[1] * g_fft_output[1];
      else { float re = g_fft_output[2*k], im = g_fft_output[2*k+1]; p = re*re + im*im; }
      float w = (float)(r - k) / (float)(r - c);
      e += w * p;
    }
    float hz_l = (float)l * kSampleRate / kNfft;
    float hz_r = (float)r * kSampleRate / kNfft;
    float norm = 2.0f / (hz_r - hz_l + 1e-8f);
    g_mel[m - 1] = 10.0f * log10f(fmaxf(e * norm, 1e-10f));
  }
}

// ================= DCT-II orthogonal =================
void dct() {
  for (int i = 0; i < kNMfcc; i++) {
    float sum = 0.0f;
    for (int j = 0; j < kNMfcc; j++)
      sum += g_mel[j] * cosf(PI * i * (2 * j + 1) / (2.0f * kNMfcc));
    float norm = (i == 0) ? sqrtf(1.0f / kNMfcc) : sqrtf(2.0f / kNMfcc);
    g_mfcc[i] = sum * norm;
  }
}

// ================= FEATURE EXTRACTION + QUANTIZATION =================
void extract_and_quantize() {
  const float   scale = g_input->params.scale;
  const int32_t zp    = g_input->params.zero_point;
  for (int f = 0; f < kNumFrames; f++) {
    compute_fft(f * kHopLength);
    mel_filter();
    dct();
    for (int c = 0; c < kNMfcc; c++) {
      float normalized = (g_mfcc[c] - kMfccMean) / (kMfccStd + 1e-8f);
      int32_t q = (int32_t)(normalized / scale) + zp;
      if (q >  127) q =  127;
      if (q < -128) q = -128;
      g_input->data.int8[c * kNumFrames + f] = (int8_t)q;
    }
  }
}

// ================= TFLM INIT =================
bool setup_tflm() {
  static uint8_t tensor_arena[130 * 1024];
  g_model = tflite::GetModel(nepspot_model_data);

  static tflite::MicroMutableOpResolver<26> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddMean();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddSpaceToBatchNd();
  resolver.AddBatchToSpaceNd();
  resolver.AddPad();
  resolver.AddPadV2();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddStridedSlice();
  resolver.AddConcatenation();
  resolver.AddSplit();
  resolver.AddTranspose();
  resolver.AddGather();
  resolver.AddShape();
  resolver.AddLogistic();

  static tflite::MicroInterpreter interpreter(
    g_model, resolver, tensor_arena, sizeof(tensor_arena));
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("TFLM alloc failed");
    return false;
  }

  g_input  = g_interpreter->input(0);
  g_output = g_interpreter->output(0);

  Serial.println("[TFLM] Backend: Chirale_TensorFlowLite + CMSIS-NN int8 kernels");
  Serial.println("[TFLM] Model: DS-CNN 12-class phase1");
  Serial.print("Input scale: ");      Serial.println(g_input->params.scale, 6);
  Serial.print("Input zero_point: "); Serial.println(g_input->params.zero_point);

  return true;
}

} // namespace

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("\nNepSpot booting...");

  init_mel();
  arm_rfft_fast_init_f32(&g_fft_instance, kNfft);

  PDM.onReceive(onPDMdata);
  PDM.begin(1, kSampleRate);
  PDM.setGain(40);

  if (!setup_tflm()) {
    Serial.println("TFLM init failed — halting");
    while (1);
  }

  Serial.println("Ready! First word in:");
  countdown();
}

// ================= LOOP =================
void loop() {
  if (!g_audioReady) return;

  // RMS silence gate — scaled for int16
  float rmsSum = 0.0f;
  for (int i = 0; i < kAudioSamples; i++) {
    float s = g_audioBuffer[i] / 32768.0f;
    rmsSum += s * s;
  }
  float rms = sqrtf(rmsSum / kAudioSamples);

  if (rms < kRmsThreshold) {
    Serial.println("[Too quiet — speak louder or move closer]");
    Serial.println("Try again:");
    countdown();
    return;
  }

  extract_and_quantize();

  uint32_t t0 = micros();
  g_interpreter->Invoke();
  uint32_t t1 = micros();
  Serial.print("[Inference: "); Serial.print(t1 - t0); Serial.println(" us]");

  int    best    = 0;
  int8_t bestVal = -128;

  Serial.print("Scores: ");
  for (int i = 0; i < kKeywordCount; i++) {
    int8_t v = g_output->data.int8[i];
    if (v > bestVal) { bestVal = v; best = i; }
    Serial.print(kKeywords[i]); Serial.print(":"); Serial.print(v); Serial.print(" ");
  }
  Serial.println();

  Serial.print(">>> Heard: ");
  Serial.println(kKeywords[best]);
  if (bestVal > 40) Serial.println(">>> DETECTED");

  Serial.println("---");
  Serial.println("Next word:");
  countdown();
}