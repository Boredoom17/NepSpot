import tensorflow as tf
from tensorflow.keras import layers, models


def build_vanilla_cnn(input_shape=(40, 32, 1), num_classes=12):
    """
    Vanilla CNN baseline for Nepali keyword spotting.

    This matches the DS-CNN depth and feature map counts, but uses standard
    2D convolutions instead of depthwise separable blocks.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv2D(64, kernel_size=(10, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='Vanilla_CNN_NepSpot')
    return model


if __name__ == "__main__":
    model = build_vanilla_cnn()
    model.summary()