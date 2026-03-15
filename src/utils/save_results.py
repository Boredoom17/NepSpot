import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.dataset import KEYWORDS, MODELS_DIR, RESULTS_DIR, ensure_directory, load_split_dataset, summarize_split

plt.rcParams['axes.unicode_minus'] = False

# ── FIXED Font setup — Clean and reliable ──
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Devanagari MT', 'Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']

print("Font family set to:", plt.rcParams['font.family'])

NEPALI_LABELS = {
    'baalnu': 'बाल्नु', 'banda': 'बन्द', 'suru': 'सुरु',
    'roknu': 'रोक्नु', 'maathi': 'माथि', 'tala': 'तल',
    'arko': 'अर्को', 'aghillo': 'अघिल्लो', 'feri': 'फेरि',
    'thik_chha': 'ठीक छ', 'huncha': 'हुन्छ', 'hoina': 'होइन'
}

# ── ROMANIZED FALLBACK  ──
ROMANIZED_LABELS = {
    'baalnu': 'Baalnu', 'banda': 'Banda', 'suru': 'Suru',
    'roknu': 'Roknu', 'maathi': 'Maathi', 'tala': 'Tala',
    'arko': 'Arko', 'aghillo': 'Aghillo', 'feri': 'Feri',
    'thik_chha': 'Thik Chha', 'huncha': 'Huncha', 'hoina': 'Hoina'
}

def get_plot_labels(le_classes):
    """Smart label selection: Devanagari if font works, else Romanized"""
    try:
        # Test if Devanagari renders (try first character)
        test_label = NEPALI_LABELS[le_classes[0]]
        fig, ax = plt.subplots(figsize=(1,1))
        ax.text(0.5, 0.5, test_label, fontsize=12)
        fig.canvas.draw()
        # If no error and renders, use Devanagari
        plt.close(fig)
        return [NEPALI_LABELS[c] for c in le_classes]
    except:
        print("Devanagari font test failed, using Romanized labels")
        return [ROMANIZED_LABELS[c] for c in le_classes]

def main():
    ensure_directory(f"{RESULTS_DIR}/figures")
    ensure_directory(f"{RESULTS_DIR}/metrics")

    # ── Load data ──
    print("Loading speaker-independent test split...")
    X, y, speaker_ids, test_speakers = load_split_dataset('test')
    X = X[..., np.newaxis]

    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(MODELS_DIR, 'label_classes.npy'), allow_pickle=True)
    y_encoded = le.transform(y)
    test_summary = summarize_split('test')
    X_test = X
    y_test = y_encoded

    # ── Load model ──
    print("Loading model...")
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))

    # ── Predict ──
    print("Running predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = np.mean(y_pred == y_test) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # ── Classification report ──
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print("\nClassification Report:")
    print(report)

    with open(f"{RESULTS_DIR}/metrics/classification_report.txt", 'w') as f:
        f.write("Evaluation split: speaker-independent test\n")
        f.write("Speakers: " + ", ".join(test_speakers) + "\n")
        f.write("Samples: " + str(test_summary['num_samples']) + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    print("Saved: results/metrics/classification_report.txt")

    # Smart label selection 
    plot_labels = get_plot_labels(le.classes_)
    print(f"Using labels: {plot_labels[:2]}...")  # Show sample

    # ── Confusion matrix ──
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=plot_labels,
        yticklabels=plot_labels,
        cmap='Reds'
    )
    plt.title('NepSpot — Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/confusion_matrix.png")

    # ── Per class accuracy bar chart ──
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(plot_labels)), per_class, 
                   color='#8B1A1A', edgecolor='white', width=0.7)
    plt.axhline(y=accuracy, color='gold', linestyle='--',
                linewidth=2, label=f'Overall: {accuracy:.1f}%')
    plt.title('Per-Keyword Accuracy — NepSpot', fontsize=14)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Keyword')
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.legend()
    for bar, val in zip(bars, per_class):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/per_keyword_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/figures/per_keyword_accuracy.png")

    with open(f"{RESULTS_DIR}/metrics/test_speakers.txt", 'w') as f:
        for speaker in test_speakers:
            speaker_count = int(np.sum(speaker_ids == speaker))
            f.write(f"{speaker}\t{speaker_count}\n")
    print("Saved: results/metrics/test_speakers.txt")

    print("\nAll results saved to results/ folder.")

if __name__ == "__main__":
    main()
