import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augment import augment_file, load_audio, speed_stretch
import soundfile as sf

"""Utilities to populate `data/raw/` with augmented audio.

What: fills folders to a target clip count and adds speed/pitch variants.
How: uses augmentation helpers in `augment.py` and writes WAV files.
"""

RAW_DIR = "data/raw"
TARGET  = 10
SPEED_NOISE_FACTOR = 0.004

EXCLUDED_SPEAKERS = {
    'voicer25', 'voicer26', 'voicer27',
    'voicer28', 'voicer29', 'voicer30',
    '_silence_test', '_unknown_test',
}

KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

def count_clips(folder):
    """Count how many wav files are in a folder"""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.wav') 
                and '_aug_' not in f])

def get_existing_clips(folder):
    """Get list of original wav files only"""
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f) 
        for f in os.listdir(folder) 
        if f.endswith('.wav') and '_aug_' not in f
    ])

def fill_to_target(speaker, word, folder):
    """Augment until we have TARGET clips"""
    existing = get_existing_clips(folder)
    current  = len(existing)
    
    if current == 0:
        return 0  # nothing to augment from
    
    if current >= TARGET:
        return 0  # already enough
    
    needed = TARGET - current
    print(f"    {word}: {current} clips -> need {needed} more")
    
    added = 0
    # cycle through existing clips and augment them
    i = 0
    while added < needed:
        source_clip = existing[i % len(existing)]
        
        # pick augmentation type based on how many we've added
        aug_types = ['noise', 'stretch', 'pitch', 'shift']
        aug_name  = aug_types[added % len(aug_types)]
        
        out_name  = f"aug_{str(added+1).zfill(2)}_{aug_name}.wav"
        out_path  = os.path.join(folder, out_name)
        
        try: 
            import numpy as np
            import librosa
            audio, sr = load_audio(source_clip)
            
            if aug_name == 'noise':
                augmented = audio + 0.005 * np.random.randn(len(audio))
            elif aug_name == 'stretch':
                augmented = librosa.effects.time_stretch(
                    audio, rate=np.random.uniform(0.85, 1.15))
            elif aug_name == 'pitch':
                augmented = librosa.effects.pitch_shift(
                    audio, sr=sr, n_steps=np.random.uniform(-2, 2))
            elif aug_name == 'shift':
                shift = int(np.random.uniform(-0.1, 0.1) * len(audio))
                augmented = np.roll(audio, shift)
            
            sf.write(out_path, augmented, sr)
            added += 1
        except Exception as e:
            print(f"      aug failed: {e}")
        
        i += 1
    
    return added

def augment_for_speed(speaker, word, folder):
    """Create slow, slow_noisy, fast, fast_noisy versions of every real clip."""
    import numpy as np

    real_clips = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith('.wav') and 'aug' not in f
    ])

    variants = [
        ('slow', False, 'aug_slow'),
        ('slow', True,  'aug_slow_noisy'),
        ('fast', False, 'aug_fast'),
        ('fast', True,  'aug_fast_noisy'),
    ]

    added = 0
    for clip_path in real_clips:
        basename = os.path.splitext(os.path.basename(clip_path))[0]
        out_paths = {
            suffix: os.path.join(folder, f"{basename}_{suffix}.wav")
            for _, _, suffix in variants
        }

        if all(os.path.exists(p) for p in out_paths.values()):
            continue

        try:
            audio, sr = load_audio(clip_path)
            for mode, add_noise_flag, suffix in variants:
                out_path = out_paths[suffix]
                if os.path.exists(out_path):
                    continue
                stretched = speed_stretch(audio, mode=mode)
                if add_noise_flag:
                    stretched = stretched + SPEED_NOISE_FACTOR * np.random.randn(len(stretched))
                sf.write(out_path, stretched, sr)
                added += 1
        except Exception as e:
            print(f"      {e}")

    return added

def main():
    speakers = sorted(os.listdir(RAW_DIR))
    speakers = [s for s in speakers if os.path.isdir(os.path.join(RAW_DIR, s))]
    
    print(f"Scanning {len(speakers)} speakers...\n")
    
    total_added = 0
    needs_aug   = []

    # Pass 1 — fill incomplete speakers to TARGET
    for speaker in speakers:
        for word in KEYWORDS:
            folder = os.path.join(RAW_DIR, speaker, word)
            count  = count_clips(folder)
            if 0 < count < TARGET:
                needs_aug.append((speaker, word, folder, count))

    if needs_aug:
        print(f"Found {len(needs_aug)} folders needing filling:\n")
        for speaker, word, folder, count in needs_aug:
            name = speaker.split('_', 1)[1] if '_' in speaker else speaker
            print(f"  {name}")
            total_added += fill_to_target(speaker, word, folder)

    # Pass 2 — speed augment ALL real clips (training speakers only)
    print(f"\nAdding slow + fast speed augmentation to training speakers...")
    print(f"Skipping val/test: {sorted(EXCLUDED_SPEAKERS)}")
    for speaker in speakers:
        if speaker in EXCLUDED_SPEAKERS:
            print(f"  skipping {speaker} (val/test)")
            continue
        for word in KEYWORDS:
            folder = os.path.join(RAW_DIR, speaker, word)
            if not os.path.exists(folder):
                continue
            total_added += augment_for_speed(speaker, word, folder)
        name = speaker.split('_', 1)[1] if '_' in speaker else speaker
        print(f"  OK {name}")

    print(f"\nDone. Added {total_added} augmented clips total.")

if __name__ == "__main__":
    main()