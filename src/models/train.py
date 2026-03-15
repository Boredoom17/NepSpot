import os
os.environ['PYTHONHASHSEED'] = '42'
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ds_cnn import build_ds_cnn
from utils.dataset import KEYWORDS, MODELS_DIR, ensure_directory, load_split_dataset, summarize_split


def main():
    train_summary = summarize_split('train')
    val_summary = summarize_split('val')
    test_summary = summarize_split('test')

    print('Training speakers: ' + str(train_summary['num_speakers']))
    print('Validation speakers: ' + str(val_summary['num_speakers']))
    print('Test speakers: ' + str(test_summary['num_speakers']))

    X_train, y_train, _, train_speakers = load_split_dataset('train')
    X_val, y_val, _, val_speakers = load_split_dataset('val')
    X_test, y_test, _, test_speakers = load_split_dataset('test')

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    le = LabelEncoder()
    le.fit(KEYWORDS)
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    print("Classes: " + str(list(le.classes_)))
    print('Train speakers: ' + ', '.join(train_speakers))
    print('Val speakers: ' + ', '.join(val_speakers))
    print('Test speakers: ' + ', '.join(test_speakers))
    print("Train: " + str(len(X_train)) + " samples")
    print("Val: " + str(len(X_val)) + " samples")
    print("Test: " + str(len(X_test)) + " samples")

    model = build_ds_cnn(
        input_shape=(40, 32, 1),
        num_classes=len(KEYWORDS)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    ensure_directory(MODELS_DIR)
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
        X_train, y_train_encoded,
        validation_data=(X_val, y_val_encoded),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
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