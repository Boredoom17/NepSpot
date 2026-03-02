import os
os.environ['PYTHONHASHSEED'] = '42'
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ds_cnn import build_ds_cnn

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models/saved"
KEYWORDS = [
    'baalnu', 'banda', 'suru', 'roknu',
    'maathi', 'tala', 'arko', 'aghillo',
    'feri', 'thik_chha', 'huncha', 'hoina'
]

def load_dataset():
    X = []
    y = []

    speakers = sorted([
        s for s in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, s))
    ])

    print("Loading data from " + str(len(speakers)) + " speakers...")

    for speaker in speakers:
        for word in KEYWORDS:
            folder = os.path.join(PROCESSED_DIR, speaker, word)
            if not os.path.exists(folder):
                continue
            for f in os.listdir(folder):
                if not f.endswith('.npy'):
                    continue
                path = os.path.join(folder, f)
                try:
                    mfcc = np.load(path)
                    X.append(mfcc)
                    y.append(word)
                except Exception as e:
                    print("Failed to load " + path + ": " + str(e))

    X = np.array(X)
    y = np.array(y)
    print("Loaded " + str(len(X)) + " samples")
    return X, y


def main():
    X, y = load_dataset()
    X = X[..., np.newaxis]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Classes: " + str(list(le.classes_)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print("Train: " + str(len(X_train)) + " samples, Test: " + str(len(X_test)) + " samples")

    model = build_ds_cnn(
        input_shape=(40, 32, 1),
        num_classes=len(KEYWORDS)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    print("\nStarting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy: " + str(round(accuracy * 100, 2)) + "%")
    print("Test loss: " + str(round(loss, 4)))

    np.save(os.path.join(MODELS_DIR, 'label_classes.npy'), le.classes_)
    print("\nModel saved to: " + MODELS_DIR + "/best_model.keras")
    print("Labels saved to: " + MODELS_DIR + "/label_classes.npy")

    # Save as SavedModel format for TFLite conversion
    saved_model_path = os.path.join(MODELS_DIR, 'saved_model')
    model.export(saved_model_path)
    print("SavedModel exported to: " + saved_model_path)

if __name__ == "__main__":
    main()