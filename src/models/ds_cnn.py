import tensorflow as tf
from tensorflow.keras import layers, models


def ds_block(x, filters):
    """Proper Depthwise Separable block: DW → BN → ReLU → PW → BN → ReLU"""
    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Pointwise
    x = layers.Conv2D(filters, kernel_size=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_ds_cnn(input_shape=(40, 32, 1), num_classes=12):
    """
    DS-CNN for Nepali Keyword Spotting — NepSpot
    Target: Arduino Nano 33 BLE Sense
    Input:  (40 MFCC coeffs, 32 time frames, 1 channel)
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1: Standard Conv (entry block)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Blocks 2-4: Proper Depthwise Separable blocks
    x = ds_block(x, 64)
    x = ds_block(x, 64)
    x = ds_block(x, 64)

    # Global average pooling + dropout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='DS_CNN_NepSpot')
    return model


if __name__ == "__main__":
    model = build_ds_cnn()
    model.summary()