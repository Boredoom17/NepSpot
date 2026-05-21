import pandas as pd
from collections import defaultdict

df = pd.read_csv(
    "/Users/ad/codes/NepSpot/data/external/opensir/utt_spk_text.tsv",
    sep="\t",
    header=None,
    names=["file_id", "speaker_id", "text"]
)

keywords = {
    "aghillo":   "अघिल्लो",
    "arko":      "अर्को",
    "baalnu":    "बाल्नु",
    "banda":     "बन्द",
    "roknu":     "रोक्नु",
    "thik_chha": "ठिक छ",
    "huncha":    "हुन्छ",
    "tala":      "तल",
    "maathi":    "माथि",
    "hoina":     "होइन",
    "feri":      "फेरि",
    "suru":      "सुरु",
}

print(f"Total rows: {len(df)}\n")
print(f"{'Keyword':<12} {'Devanagari':<12} {'Total clips':<14} {'Unique speakers'}")
print("-" * 55)

for eng, nep in keywords.items():
    matched = df[df["text"].str.contains(nep, na=False)]
    unique_spk = matched["speaker_id"].nunique()
    print(f"{eng:<12} {nep:<12} {len(matched):<14} {unique_spk}")
