import numpy as np
import sounddevice as sd
import tensorflow as tf
import os
import librosa

# ── Config ──
SAMPLE_RATE = 16000
DURATION    = 1.5
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 1024
MODELS_DIR  = "models/saved"
THRESHOLD   = 0.80  # minimum confidence to report
MARGIN      = 0.35  # minimum gap between top 2 predictions
SILENCE_RMS = 0.002  # below this = silence

KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

NEPALI = {
    'baalnu': 'बाल्नु', 'banda': 'बन्द', 'suru': 'सुरु',
    'roknu': 'रोक्नु', 'maathi': 'माथि', 'tala': 'तल',
    'arko': 'अर्को', 'aghillo': 'अघिल्लो', 'feri': 'फेरि',
    'thik_chha': 'ठीक छ', 'huncha': 'हुन्छ', 'hoina': 'होइन'
}

def load_model_and_labels():
    print("Loading model...")
    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'best_model.keras')
    )
    classes = np.load(
        os.path.join(MODELS_DIR, 'label_classes.npy'),
        allow_pickle=True
    )
    mean = np.load(os.path.join(MODELS_DIR, 'mfcc_mean.npy'))
    std  = np.load(os.path.join(MODELS_DIR, 'mfcc_std.npy'))
    print(f"Model loaded. Classes: {list(classes)}")
    return model, classes, mean, std

def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    print(f"\n🎤 Listening for {duration}s... speak now!")
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio, mean, std, sr=SAMPLE_RATE):
    target_len = int(sr * 1.0)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    mfcc = (mfcc - mean) / (std + 1e-8)
    return mfcc

def predict(model, classes, mfcc):
    x     = mfcc[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]

    # Top 2 predictions
    top2        = np.argsort(probs)[-2:][::-1]
    idx         = top2[0]
    conf        = probs[idx]
    second_conf = probs[top2[1]]
    word        = classes[idx]

    # Reject if margin between top 2 is too small
    margin = conf - second_conf
    if margin < MARGIN:
        return None, conf

    return word, conf

def main():
    model, classes, mean, std = load_model_and_labels()

    print("\n" + "="*40)
    print("NepSpot Live Mic Test")
    print("Press Enter to record, Ctrl+C to quit")
    print("="*40)

    while True:
        try:
            input("\nPress Enter to speak...")
            import time
            time.sleep(0.5)
            audio = record_audio()

            # Silence detection
            rms = np.sqrt(np.mean(audio**2))
            if rms < SILENCE_RMS:
                print("🔇 Silence detected — no keyword found")
                continue

            mfcc       = extract_mfcc(audio, mean, std)
            word, conf = predict(model, classes, mfcc)

            if word is None:
                print(f"❓ Rejected — sound not recognized as a keyword")
            elif conf >= THRESHOLD:
                nepali = NEPALI.get(word, word)
                print(f"✅ Detected: {word} ({nepali}) — {conf*100:.1f}% confidence")
            else:
                print(f"❓ Uncertain: {word} — only {conf*100:.1f}% confidence")

        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()