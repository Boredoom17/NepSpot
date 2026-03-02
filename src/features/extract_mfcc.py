import os
import numpy as np
import librosa

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
STATS_DIR     = "models/saved"
SAMPLE_RATE   = 16000
DURATION      = 1.0
N_MFCC        = 40
HOP_LENGTH    = 512
N_FFT         = 1024

KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

def load_and_pad(file_path, sr=SAMPLE_RATE, duration=DURATION):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    return audio

def extract_mfcc_raw(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    return mfcc

def collect_all_mfccs():
    all_mfccs = []
    speakers = sorted([
        s for s in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, s))
    ])
    print("Pass 1/2 — Collecting raw MFCCs from " + str(len(speakers)) + " speakers...")
    for speaker in speakers:
        for word in KEYWORDS:
            folder = os.path.join(RAW_DIR, speaker, word)
            if not os.path.exists(folder):
                continue
            for wav_file in sorted(os.listdir(folder)):
                if not wav_file.endswith('.wav'):
                    continue
                try:
                    audio = load_and_pad(os.path.join(folder, wav_file))
                    mfcc = extract_mfcc_raw(audio)
                    all_mfccs.append(mfcc)
                except Exception as e:
                    print("  Failed: " + str(e))
    return np.array(all_mfccs)

def compute_and_save_stats(all_mfccs):
    global_mean = np.mean(all_mfccs)
    global_std  = np.std(all_mfccs)
    print("Global stats — mean: " + str(round(float(global_mean), 4)) + ", std: " + str(round(float(global_std), 4)))
    os.makedirs(STATS_DIR, exist_ok=True)
    np.save(os.path.join(STATS_DIR, 'mfcc_mean.npy'), global_mean)
    np.save(os.path.join(STATS_DIR, 'mfcc_std.npy'),  global_std)
    return global_mean, global_std

def save_normalized(all_mfccs, global_mean, global_std):
    speakers = sorted([
        s for s in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, s))
    ])
    print("Pass 2/2 — Normalizing and saving to " + PROCESSED_DIR + " ...")
    idx = 0
    total = 0
    for speaker in speakers:
        for word in KEYWORDS:
            in_folder  = os.path.join(RAW_DIR, speaker, word)
            out_folder = os.path.join(PROCESSED_DIR, speaker, word)
            if not os.path.exists(in_folder):
                continue
            os.makedirs(out_folder, exist_ok=True)
            for wav_file in sorted(os.listdir(in_folder)):
                if not wav_file.endswith('.wav'):
                    continue
                out_path = os.path.join(out_folder, wav_file.replace('.wav', '.npy'))
                mfcc_normalized = (all_mfccs[idx] - global_mean) / (global_std + 1e-8)
                np.save(out_path, mfcc_normalized)
                idx += 1
                total += 1
        name = speaker.split('_', 1)[1] if '_' in speaker else speaker
        print("  ✓ " + name)
    print("Done. Saved " + str(total) + " normalized files.")

def process_all():
    all_mfccs = collect_all_mfccs()
    print("Total clips: " + str(len(all_mfccs)))
    global_mean, global_std = compute_and_save_stats(all_mfccs)
    save_normalized(all_mfccs, global_mean, global_std)
    print("Stats saved to: " + STATS_DIR)
    print("Use mfcc_mean.npy and mfcc_std.npy at inference time on Arduino too.")

if __name__ == "__main__":
    process_all()