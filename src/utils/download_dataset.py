import cloudinary
import cloudinary.api
import urllib.request
import os
import json
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD"),
    api_key=os.getenv("CLOUDINARY_KEY"),
    api_secret=os.getenv("CLOUDINARY_SECRET")
)

DOWNLOAD_DIR = "data/raw"
CONFIG_PATH  = "configs/speaker_split_v1.json"


def get_speakers_and_keywords():
    """Read speaker list and keywords from config — avoids stale Cloudinary folder cache."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    splits = config["splits"]
    speakers = (
        splits["train_complete"] +
        splits["train_partial"] +
        splits["val"] +
        splits["test"]
    )
    return speakers, config["keywords"]


def get_clips_for_speaker_word(speaker, word):
    folder = f"recordings/{speaker}/{word}"
    try:
        result = cloudinary.api.resources(
            resource_type="video",
            type="upload",
            prefix=folder,
            max_results=50
        )
        return result.get("resources", [])
    except Exception:
        return []


def download_and_convert(url, save_path):
    """Download webm and convert to 16kHz mono wav."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = save_path.replace(".wav", "_temp.webm")
    urllib.request.urlretrieve(url, temp_path)
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(save_path, format="wav")
    os.remove(temp_path)


def main():
    speakers, keywords = get_speakers_and_keywords()
    print(f"Speakers : {len(speakers)}")
    print(f"Keywords : {len(keywords)}")
    print(f"Output   : {DOWNLOAD_DIR}\n")

    total, failed = 0, 0

    for speaker in speakers:
        print(f"Downloading: {speaker}")
        for word in keywords:
            clips = get_clips_for_speaker_word(speaker, word)
            for i, clip in enumerate(clips):
                save_path = os.path.join(
                    DOWNLOAD_DIR, speaker, word,
                    f"attempt_{str(i+1).zfill(2)}.wav"
                )
                if os.path.exists(save_path):
                    continue
                try:
                    download_and_convert(clip["secure_url"], save_path)
                    total += 1
                except Exception as e:
                    print(f"  ✗ {word} attempt {i+1}: {e}")
                    failed += 1
        print(f"  ✓ done")

    print(f"\nTotal downloaded : {total}")
    print(f"Failed           : {failed}")
    print(f"Saved to         : {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()