# One Speaker Workflow

## Your Data Structure

**Original NepSpot data** (keep as-is):
```
data/raw/
  voicer1/
  voicer2/
  ... (your 30 speakers)
```

**Downloaded Common Voice clips**:
```
data/external/staging/raw_clips/
  cv_...huncha.wav
  cv_...feri.wav
  ... (119 full sentence clips)
```

**Working folder** (one speaker at a time):
```
data/external/working/current_speaker/
  cv_<speaker1>/
    (10-20 full sentence clips for this speaker)
```

## Simple Workflow

1. Run one command:
   ```bash
   python3 scripts/data_mining/one_speaker_workflow.py
   ```

2. Choose a speaker (1-30)

3. It copies their 10-20 clips to a single working folder

4. Open Audacity:
   - File → Open → Point to that folder
   - Trim each clip
   - Export to: `data/raw/cv_<speaker>/<keyword>/clip.wav`

5. Repeat for next speaker

## Audacity Quick Guide

1. Open clip
2. Spacebar to play
3. Click-drag to select the keyword portion
4. Cmd+I to invert (select everything EXCEPT keyword)
5. Cmd+X to delete the rest
6. Cmd+Shift+E to export
7. Save as WAV to: `data/raw/cv_<speaker>/huncha/filename.wav`

Done!
