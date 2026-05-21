# Clean Workflow: One Speaker at a Time

## Your Data is Now Separated

**Original NepSpot** (voicer1-30): Untouched in `data/raw/`

**Downloaded Common Voice** (119 clips): In `data/external/staging/raw_clips/`

**Working folder**: `data/external/working/current_speaker/`

## How to Use

### Step 1: Start the workflow

```bash
python3 scripts/data_mining/one_speaker_workflow.py
```

Output:
```
Found 119 clips total from Common Voice
Speakers: 23
  1. cv_008ec3f33e366622 (11 clips)
  2. cv_0142a5d130e7fb1d (12 clips)
  3. cv_13b830feb8a132fa (20 clips)
  ... etc
```

### Step 2: Pick a speaker

Enter number (e.g., `1` for the first speaker with 11 clips)

The script will:
- Copy those 11 clips to: `data/external/working/current_speaker/cv_008ec3.../`
- Create keyword folders in `data/raw/cv_008ec3.../huncha/`, `data/raw/cv_008ec3.../feri/`, etc.
- Show you the path

### Step 3: Open Audacity

1. File → Open
2. Navigate to the working folder it showed you
3. Open first clip

### Step 4: Trim and Export (repeat for each clip)

**In Audacity:**
1. **Spacebar** = Play
2. **Click + drag** on waveform to select keyword portion (the 1-second part you want)
3. **Cmd+I** = Invert (select everything EXCEPT the keyword)
4. **Cmd+X** = Delete the rest (now you have only the keyword)
5. **Cmd+Shift+E** = Export as WAV
6. Save to: `data/raw/cv_008ec3.../huncha/clip_name.wav`
7. **Cmd+W** = Close and open next clip

### Step 5: Repeat for next speaker

1. Go back to terminal
2. Quit previous run (Ctrl+C)
3. Run again: `python3 scripts/data_mining/one_speaker_workflow.py`
4. Pick next speaker

## Example Path

**Original speaker with 11 clips:**
```
Working folder:
  /data/external/working/current_speaker/cv_008ec3.../
    clip1.wav (full sentence)
    clip2.wav (full sentence)
    ... (11 total)

After trimming in Audacity, save to:
  data/raw/cv_008ec3.../huncha/clip1.wav
  data/raw/cv_008ec3.../feri/clip2.wav
  data/raw/cv_008ec3.../huncha/clip3.wav
  ... etc
```

## After Trimming All Speakers

```bash
python src/features/extract_mfcc.py
python src/models/train_bcresnet.py
```

Done!
