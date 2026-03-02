import cloudinary
import cloudinary.api
import urllib.request
import os
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD"),
    api_key    = os.getenv("CLOUDINARY_KEY"),
    api_secret = os.getenv("CLOUDINARY_SECRET")
)

DOWNLOAD_DIR = "data/raw"

def get_all_speakers():
    result = cloudinary.api.subfolders("recordings")
    return [f["name"] for f in result["folders"]]

def get_clips_for_speaker_word(speaker, word):
    folder = f"recordings/{speaker}/{word}"
    try:
        result = cloudinary.api.resources(
            type="upload",
            prefix=folder,
            resource_type="video",
            max_results=50
        )
        return result.get("resources", [])
    except:
        return []

def download_and_convert(url, save_path):
    """Download webm and convert to 16kHz mono wav"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download to temp file
    temp_path = save_path.replace(".wav", "_temp.webm")
    urllib.request.urlretrieve(url, temp_path)
    
    # Convert to wav 16kHz mono
    audio = AudioSegment.from_file(temp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(save_path, format="wav")
    
    # Delete temp file
    os.remove(temp_path)

def main():
    keywords = [
        'baalnu', 'banda', 'suru', 'roknu',
        'maathi', 'tala', 'arko', 'aghillo',
        'feri', 'thik_chha', 'huncha', 'hoina'
    ]

    print("Fetching speaker list...")
    speakers = get_all_speakers()
    print(f"Found {len(speakers)} speakers\n")

    total = 0

    for speaker in speakers:
        print(f"Downloading: {speaker}")
        for word in keywords:
            clips = get_clips_for_speaker_word(speaker, word)
            for i, clip in enumerate(clips):
                url       = clip["secure_url"]
                save_path = os.path.join(
                    DOWNLOAD_DIR, speaker, word,
                    f"attempt_{str(i+1).zfill(2)}.wav"
                )
                # Skip if already downloaded
                """if os.path.exists(save_path):
                    continue"""
                try:
                    download_and_convert(url, save_path)
                    total += 1
                except Exception as e:
                    print(f"  ✗ Failed {word} attempt {i+1}: {e}")
        print(f"  ✓ done")

    print(f"\nTotal downloaded: {total} clips")
    print(f"Saved to: {DOWNLOAD_DIR}")

if __name__ == "__main__":
    main()