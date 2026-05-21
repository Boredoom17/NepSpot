import pandas as pd
import whisper
import os
import zipfile
from pydub import AudioSegment

"""Extract targeted Nepali keyword clips from OpenSLR ASR zips.

What: scans an OpenSLR TSV and ASR zipfiles, locates target keywords via
Whisper transcriptions, and exports short centered clips for each match.

How: selects one clip per speaker per keyword up to `TARGET`, runs
Whisper medium for word timestamps, centers a `CLIP_MS` window on the
detected word, and writes 16 kHz mono WAV files under `OUT_BASE`.
"""

# config
TSV_PATH = "/Users/ad/codes/NepSpot/data/external/opensir/utt_spk_text.tsv"
ZIPS_DIR = "/Users/ad/codes/NepSpot/data/external/opensir/zips"
OUT_BASE = "/Users/ad/codes/NepSpot/data/raw"
TEMP_DIR = "/Users/ad/codes/NepSpot/data/external/opensir/temp"
TARGET = 400
CLIP_MS = 1000

KEYWORDS = {
    "aghillo":   "अघिल्लो",
    "arko":      "अर्को",
    "baalnu":    "बाल्नु",
    "banda":     "बन्द",
    "roknu":     "रोक्नु",
    "thik_chha": "ठिक छ",
    "tala":      "तल",
    "maathi":    "माथि",
    "hoina":     "होइन",
    "feri":      "फेरि",
    "suru":      "सुरु",
}

CURRENT = {
    "aghillo": 240, "arko": 231, "baalnu": 270, "banda": 270,
    "roknu": 240, "thik_chha": 280, "tala": 235, "maathi": 280,
    "hoina": 250, "feri": 320, "suru": 340,
}

os.makedirs(TEMP_DIR, exist_ok=True)

print("Loading TSV...")
df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["file_id", "speaker_id", "text"])

# Build selection: 1 clip per speaker per keyword
selected = {}
for eng, nep in KEYWORDS.items():
    need = max(0, TARGET - CURRENT[eng])
    if need == 0:
        selected[eng] = []
        continue
    matched = df[df["text"].str.contains(nep, na=False)].drop_duplicates("speaker_id")
    taking = matched.head(need)
    selected[eng] = list(zip(taking["file_id"], taking["speaker_id"]))
    print(f"{eng}: need {need}, taking {len(taking)}")

# Build reverse map: file_id -> keywords
file_to_keywords = {}
for eng, items in selected.items():
    for file_id, spk_id in items:
        if file_id not in file_to_keywords:
            file_to_keywords[file_id] = []
        file_to_keywords[file_id].append(eng)

print(f"Total unique files to process: {len(file_to_keywords)}")

# Load Whisper medium
print("Loading Whisper medium model...")
model = whisper.load_model("medium")

saved_count = {eng: 0 for eng in KEYWORDS}
failed_count = {eng: 0 for eng in KEYWORDS}

zip_names = [f"asr_nepali_{p}.zip" for p in "0123456789abcdef"]

for zip_name in zip_names:
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    if not os.path.exists(zip_path):
        print(f"Skipping {zip_name} (not found)")
        continue

    print(f"Processing {zip_name}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = zf.namelist()

        for file_id, keywords_needed in file_to_keywords.items():
            # find flac inside zip
            matches = [n for n in all_names if file_id in n and n.endswith(".flac")]
            if not matches:
                continue

            flac_name = matches[0]
            temp_flac = os.path.join(TEMP_DIR, f"{file_id}.flac")

            # extract to temp
            with zf.open(flac_name) as src, open(temp_flac, "wb") as dst:
                dst.write(src.read())

            # run Whisper
            try:
                result = model.transcribe(temp_flac, language="ne", word_timestamps=True)
            except Exception as e:
                print(f"Whisper failed for {file_id}: {e}")
                os.remove(temp_flac)
                continue

            # load audio for cutting
            audio = AudioSegment.from_file(temp_flac, format="flac")
            audio = audio.set_frame_rate(16000).set_channels(1)

            for eng in keywords_needed:
                nep = KEYWORDS[eng]
                found = False
                for seg in result.get("segments", []):
                    for word_info in seg.get("words", []):
                        word_text = word_info.get("word", "").strip()
                        if nep in word_text or word_text in nep:
                            start_ms = int(word_info["start"] * 1000)
                            end_ms   = int(word_info["end"] * 1000)
                            mid_ms   = (start_ms + end_ms) // 2
                            clip_start = max(0, mid_ms - CLIP_MS // 2)
                            clip_end   = clip_start + CLIP_MS

                            clip = audio[clip_start:clip_end]

                            out_dir = os.path.join(OUT_BASE, f"openslr_{file_id[:4]}", eng)
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"{file_id}.wav")
                            clip.export(out_path, format="wav", parameters=["-ar", "16000"])
                            saved_count[eng] += 1
                            print(f"Saved {eng} [{saved_count[eng]}] {file_id}")
                            found = True
                            break
                    if found:
                        break

                if not found:
                    failed_count[eng] += 1
                    print(f"Not found: {eng} in {file_id}")

            os.remove(temp_flac)

# cleanup temp dir
try:
    os.rmdir(TEMP_DIR)
except OSError:
    pass

print("Final counts:")
for eng in KEYWORDS:
    total = CURRENT[eng] + saved_count[eng]
    print(f"{eng:<12} +{saved_count[eng]:<5} (failed:{failed_count[eng]}) -> total {total}")

print("Done. Consider removing the zips directory to free space:")
print(f"  rm -rf {ZIPS_DIR}")
