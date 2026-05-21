import pandas as pd
import os
import zipfile
import io
from pydub import AudioSegment

"""Organize and extract SLR clips into a speaker-structured folder layout.

What: selects files containing target Nepali keywords, maps speakers to
sequential speaker IDs, and writes fixed-length WAV excerpts organized as
`OUT_BASE/speaker_XXX/<keyword>/*.wav`.

How: estimates the keyword's position in the ASR text, centers a `CLIP_MS`
window on that position, pads if necessary, and exports a 16 kHz mono clip.
"""

# config
TSV_PATH = "/Users/ad/codes/NepSpot/data/external/opensir/utt_spk_text.tsv"
ZIPS_DIR = "/Users/ad/codes/NepSpot/data/external/opensir/zips"
OUT_BASE = "/Users/ad/codes/NepSpot/data/external/slr_db"
TARGET = 400
CLIP_MS = 2000

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

print("Loading TSV...")
df = pd.read_csv(TSV_PATH, sep="\t", header=None, names=["file_id", "speaker_id", "text"])

file_to_info = {}
speaker_number = {}
spk_counter = [1]

for eng, nep in KEYWORDS.items():
    need = max(0, TARGET - CURRENT[eng])
    if need == 0:
        continue
    matched = df[df["text"].str.contains(nep, na=False)].drop_duplicates("speaker_id")
    taking = matched.head(need)
    for _, row in taking.iterrows():
        fid = row["file_id"]
        spk = row["speaker_id"]
        txt = row["text"]
        if spk not in speaker_number:
            speaker_number[spk] = spk_counter[0]
            spk_counter[0] += 1
        if fid not in file_to_info:
            file_to_info[fid] = {"speaker_id": spk, "keywords": {}}
        file_to_info[fid]["keywords"][eng] = txt

print(f"Unique speakers : {len(speaker_number)}")
print(f"Unique files    : {len(file_to_info)}")
print(f"Output folder   : {OUT_BASE}\n")

def estimate_cut_ms(text, nep_keyword, audio_duration_ms, clip_ms):
    """Estimate a centered clip window around the keyword occurrence.

    How: finds the token index containing `nep_keyword`, maps to a time
    proportion in the audio, and returns start/end milliseconds for a
    `clip_ms` window (pads if near edges).
    """
    words = text.strip().split()
    total = len(words)
    idx = next((i for i, w in enumerate(words) if nep_keyword in w), 0)
    mid = int((idx + 0.5) / total * audio_duration_ms)
    start = max(0, mid - clip_ms // 2)
    end = min(audio_duration_ms, start + clip_ms)
    if end - start < clip_ms:
        start = max(0, end - clip_ms)
    return start, end

zip_names = [f"asr_nepali_{p}.zip" for p in "0123456789abcdef"]
extracted = 0
skipped = 0

for zip_name in zip_names:
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    if not os.path.exists(zip_path):
        print(f"Skipping {zip_name} (not downloaded)")
        continue

    print(f"Processing {zip_name}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = zf.namelist()

        for file_id, info in file_to_info.items():
            matches = [n for n in all_names if file_id in n and n.endswith(".flac")]
            if not matches:
                continue

            spk_id = info["speaker_id"]
            spk_num = speaker_number[spk_id]
            spk_folder = f"speaker_{spk_num:03d}"

            data = zf.read(matches[0])
            audio = AudioSegment.from_file(io.BytesIO(data), format="flac")
            audio = audio.set_frame_rate(16000).set_channels(1)
            dur = len(audio)

            for eng, text in info["keywords"].items():
                out_dir = os.path.join(OUT_BASE, spk_folder, eng)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{file_id}.wav")

                if os.path.exists(out_path):
                    skipped += 1
                    continue

                nep = KEYWORDS[eng]
                start, end = estimate_cut_ms(text, nep, dur, CLIP_MS)
                clip = audio[start:end]

                if len(clip) < CLIP_MS:
                    clip = clip + AudioSegment.silent(duration=CLIP_MS - len(clip))

                clip.export(out_path, format="wav", parameters=["-ar", "16000"])
                extracted += 1

    print(f"  done ({extracted} extracted so far)")

print("Extraction complete")
print(f"Extracted : {extracted}")
print(f"Skipped   : {skipped}")
print(f"Open: {OUT_BASE}")