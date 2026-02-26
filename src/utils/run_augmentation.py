import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augment import augment_file, load_audio
import soundfile as sf

RAW_DIR = "data/raw"
TARGET  = 10

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
    print(f"    {word}: {current} clips → need {needed} more")
    
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
            print(f"      ✗ aug failed: {e}")
        
        i += 1
    
    return added

def main():
    speakers = sorted(os.listdir(RAW_DIR))
    speakers = [s for s in speakers if os.path.isdir(os.path.join(RAW_DIR, s))]
    
    print(f"Scanning {len(speakers)} speakers...\n")
    
    total_added = 0
    needs_aug   = []

    # First pass — find who needs augmentation
    for speaker in speakers:
        for word in KEYWORDS:
            folder  = os.path.join(RAW_DIR, speaker, word)
            count   = count_clips(folder)
            if 0 < count < TARGET:
                needs_aug.append((speaker, word, folder, count))

    if not needs_aug:
        print("All speakers have 10+ clips per keyword. No augmentation needed.")
        return

    print(f"Found {len(needs_aug)} keyword folders needing augmentation:\n")
    
    for speaker, word, folder, count in needs_aug:
        speaker_name = speaker.split('_', 1)[1] if '_' in speaker else speaker
        print(f"  {speaker_name}")
        added = fill_to_target(speaker, word, folder)
        total_added += added

    print(f"\nDone. Added {total_added} augmented clips total.")

if __name__ == "__main__":
    main()