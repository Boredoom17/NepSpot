import librosa
import soundfile as sf
import numpy as np
import os

def load_audio(file_path, sr=16000):
    """Load an audio file at 16kHz mono"""
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr

def add_noise(audio, noise_factor=0.005):
    """Add slight random background noise"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, rate=None):
    """Make the word slightly faster or slower"""
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, steps=None):
    """Raise or lower the pitch slightly"""
    if steps is None:
        steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def time_shift(audio, shift_max=0.1):
    """Shift audio slightly left or right in time"""
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def augment_file(input_path, output_dir, num_augments=4):
    """
    Take one audio file and create num_augments variations of it.
    Saves them in output_dir with _aug1, _aug2 etc suffix.
    """
    audio, sr = load_audio(input_path)
    
    # Get base filename without extension
    basename = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    augmentations = [
        ("noise",   lambda a: add_noise(a)),
        ("stretch", lambda a: time_stretch(a)),
        ("pitch",   lambda a: pitch_shift(a, sr)),
        ("shift",   lambda a: time_shift(a)),
    ]
    
    saved = []
    for i, (name, fn) in enumerate(augmentations[:num_augments]):
        augmented = fn(audio)
        out_path  = os.path.join(output_dir, f"{basename}_aug_{name}.wav")
        sf.write(out_path, augmented, sr)
        saved.append(out_path)
        print(f"  Saved: {os.path.basename(out_path)}")
    
    return saved

if __name__ == "__main__":
    # Quick test
    print("Augmentation module loaded successfully")