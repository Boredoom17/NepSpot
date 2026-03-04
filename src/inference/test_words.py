import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import os

# ── Config ──
SAMPLE_RATE = 16000
DURATION    = 1.5
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 1024
MODELS_DIR  = "models/saved"
THRESHOLD   = 0.80
MARGIN      = 0.35
SILENCE_RMS = 0.002

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

TRIES_PER_WORD = 5

def load_model_and_labels():
    model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, 'best_model.keras')
    )
    classes = np.load(
        os.path.join(MODELS_DIR, 'label_classes.npy'),
        allow_pickle=True
    )
    mean = np.load(os.path.join(MODELS_DIR, 'mfcc_mean.npy'))
    std  = np.load(os.path.join(MODELS_DIR, 'mfcc_std.npy'))
    return model, classes, mean, std

def record_audio():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()

def extract_mfcc(audio, mean, std):
    target_len = int(SAMPLE_RATE * 1.0)
    audio = audio[:target_len]
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    mfcc = (mfcc - mean) / (std + 1e-8)
    return mfcc

def predict(model, classes, mfcc):
    x     = mfcc[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    top2  = np.argsort(probs)[-2:][::-1]
    idx   = top2[0]
    conf  = probs[idx]
    second_conf = probs[top2[1]]
    word  = classes[idx]
    margin = conf - second_conf
    if margin < MARGIN:
        return None, conf
    return word, conf

def test_word(word, model, classes, mean, std):
    nepali  = NEPALI[word]
    correct = 0
    results = []

    print(f"\n{'='*45}")
    print(f"  Testing: {word} ({nepali})")
    print(f"  Say '{nepali}' {TRIES_PER_WORD} times")
    print(f"{'='*45}")

    for i in range(TRIES_PER_WORD):
        input(f"\n  Try {i+1}/{TRIES_PER_WORD} — Press Enter then speak...")
        import time
        time.sleep(0.5)  # wait half second before recording
        print(f"  🎤 Listening...")
        audio = record_audio()

        rms = np.sqrt(np.mean(audio**2))
        if rms < SILENCE_RMS:
            print(f"  🔇 Silence — try again")
            results.append("silence")
            continue

        mfcc = extract_mfcc(audio, mean, std)
        predicted, conf = predict(model, classes, mfcc)

        if predicted is None:
            print(f"  ❓ Rejected — unclear")
            results.append("rejected")
        elif predicted == word:
            correct += 1
            print(f"  ✅ Correct! ({conf*100:.1f}%)")
            results.append("correct")
        else:
            nepali_wrong = NEPALI.get(predicted, predicted)
            print(f"  ❌ Wrong — got {predicted} ({nepali_wrong}) at {conf*100:.1f}%")
            results.append(f"wrong:{predicted}")

    score = correct / TRIES_PER_WORD * 100
    print(f"\n  Score: {correct}/{TRIES_PER_WORD} = {score:.0f}%")
    return score, results

def main():
    print("Loading model...")
    model, classes, mean, std = load_model_and_labels()

    print("\n" + "="*45)
    print("  NepSpot — Word-by-Word Test")
    print("  Each word tested 5 times")
    print("="*45)

    print("\nWhich words to test?")
    print("0. All words")
    for i, word in enumerate(KEYWORDS):
        print(f"{i+1}. {word} ({NEPALI[word]})")

    choice = input("\nEnter number (0 for all): ").strip()

    if choice == '0':
        words_to_test = KEYWORDS
    else:
        try:
            idx = int(choice) - 1
            words_to_test = [KEYWORDS[idx]]
        except:
            print("Invalid choice, testing all.")
            words_to_test = KEYWORDS

    scores = {}
    for word in words_to_test:
        score, _ = test_word(word, model, classes, mean, std)
        scores[word] = score

    # ── Summary ──
    print("\n" + "="*45)
    print("  FINAL RESULTS")
    print("="*45)
    for word, score in sorted(scores.items(), key=lambda x: x[1]):
        nepali = NEPALI[word]
        bar    = "█" * int(score // 10)
        status = "✅" if score >= 80 else "⚠️ " if score >= 60 else "❌"
        print(f"  {status} {word:12} {nepali:10} {bar} {score:.0f}%")

if __name__ == "__main__":
    main()