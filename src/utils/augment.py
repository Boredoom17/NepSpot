import librosa
import soundfile as sf
import numpy as np
import os

def load_audio(file_path, sr=16000):
    """Load an audio file at 16kHz mono.

    How: uses librosa to read and resample to `sr`, returns (audio, sr).
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr

def add_noise(audio, noise_factor=0.005):
    """Add light Gaussian noise to the audio.

    How: draws a random normal array and scales it by `noise_factor`.
    """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_stretch(audio, rate=None):
    """Randomly time-stretch audio to simulate tempo variation.

    How: picks a random `rate` in [0.85,1.15] when not provided.
    """
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)

def speed_stretch(audio, mode='fast'):
    """Time-stretch audio in either 'fast' (1.20-1.35) or 'slow' (0.80-0.90) range.

    fast → shorter clip (model sees compressed speech)
    slow → longer clip (model sees drawn-out speech)
    """
    if mode == 'fast':
        rate = np.random.uniform(1.20, 1.35)
    elif mode == 'slow':
        rate = np.random.uniform(0.80, 0.90)
    else:
        raise ValueError("mode must be 'fast' or 'slow', got " + repr(mode))
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, steps=None):
    """Shift pitch by a small number of semitones.

    How: picks `steps` in [-2,2] when not provided and applies librosa pitch shift.
    """
    if steps is None:
        steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def time_shift(audio, shift_max=0.1):
    """Circularly shift audio by a small fraction of its length.

    How: computes a random integer shift within `±shift_max * len(audio)`.
    """
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)

def augment_file(input_path, output_dir, num_augments=4):
    """
    Take one audio file and create num_augments variations of it.
    Saves them in output_dir with _aug1, _aug2 etc suffix.
    """
    audio, sr = load_audio(input_path)

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
        print(f"Saved: {os.path.basename(out_path)}")

    return saved

if __name__ == "__main__":
    print("augment.py: module loaded — run `augment_file()` to create augmentations.")