import os
import numpy as np
import librosa

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
SAMPLE_RATE   = 16000
DURATION      = 1.0        # seconds — trim/pad all clips to 1 second
N_MFCC        = 40         # number of MFCC coefficients
HOP_LENGTH    = 512
N_FFT         = 1024

KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

def load_and_pad(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio and make it exactly `duration` seconds long"""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    target_len = int(sr * duration)
    
    if len(audio) > target_len:
        # trim from center
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        # pad with zeros
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    
    return audio

def extract_mfcc(audio, sr=SAMPLE_RATE):
    """Extract 40 MFCCs from audio"""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc

def process_all():
    speakers = sorted([
        s for s in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, s))
    ])
    
    print(f"Processing {len(speakers)} speakers...\n")
    
    total     = 0
    failed    = 0
    
    for speaker in speakers:
        for word in KEYWORDS:
            in_folder  = os.path.join(RAW_DIR, speaker, word)
            out_folder = os.path.join(PROCESSED_DIR, speaker, word)
            
            if not os.path.exists(in_folder):
                continue
            
            os.makedirs(out_folder, exist_ok=True)
            
            wav_files = sorted([
                f for f in os.listdir(in_folder)
                if f.endswith('.wav')
            ])
            
            for wav_file in wav_files:
                in_path  = os.path.join(in_folder, wav_file)
                out_path = os.path.join(
                    out_folder,
                    wav_file.replace('.wav', '.npy')
                )
                
                if os.path.exists(out_path):
                    continue
                
                try:
                    audio = load_and_pad(in_path)
                    mfcc  = extract_mfcc(audio)
                    np.save(out_path, mfcc)
                    total += 1
                except Exception as e:
                    print(f"  ✗ Failed {speaker}/{word}/{wav_file}: {e}")
                    failed += 1
        
        name = speaker.split('_', 1)[1] if '_' in speaker else speaker
        print(f"  ✓ {name}")
    
    print(f"\nDone. Processed {total} files. Failed: {failed}")
    print(f"Saved to: {PROCESSED_DIR}")

if __name__ == "__main__":
    process_all()