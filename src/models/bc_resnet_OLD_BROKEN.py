"""BC-ResNet-1 for NepSpot keyword spotting.

This Keras implementation follows the Broadcasted Residual Learning paper
and the open-source Qualcomm AI Research BC-ResNet reference design:
https://github.com/Qualcomm-AI-research/bcresnet

The model is adapted to NepSpot's fixed MFCC input shape (40, 32, 1) and
keeps the layers quantization-friendly for INT8 export.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _frequency_collapse_branch(x, filters: int, name: str):
    """Collapse the frequency axis and broadcast the result back."""

    freq_dim = x.shape[1]
    if freq_dim is None:
        raise ValueError('BC-ResNet expects a static frequency dimension')

    branch = layers.AveragePooling2D(
        pool_size=(int(freq_dim), 1),
        strides=(1, 1),
        padding='valid',
        name=f'{name}_freq_pool',
    )(x)
    branch = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_conv',
    )(branch)
    branch = layers.BatchNormalization(name=f'{name}_bn')(branch)
    return branch


def _transition_block(x, filters: int, name: str):
    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_conv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.AveragePooling2D(
        pool_size=(2, 1),
        strides=(2, 1),
        padding='same',
        name=f'{name}_pool',
    )(x)
    return x


def _normal_block(x, filters: int, dilation_rate: int, name: str):
    shortcut = x

    depthwise = layers.DepthwiseConv2D(
        kernel_size=(3, 1),
        dilation_rate=(dilation_rate, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_dwconv',
    )(x)
    depthwise = layers.BatchNormalization(name=f'{name}_dw_bn')(depthwise)

    collapsed = _frequency_collapse_branch(x, filters, name=f'{name}_collapse')

    x = layers.Add(name=f'{name}_broadcast_add')([depthwise, collapsed])
    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_pwconv',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)

    if shortcut.shape[-1] == filters and shortcut.shape[1] == x.shape[1] and shortcut.shape[2] == x.shape[2]:
        x = layers.Add(name=f'{name}_residual_add')([x, shortcut])

    x = layers.ReLU(name=f'{name}_relu')(x)
    return x


def build_bc_resnet(input_shape=(40, 32, 1), num_classes=12):
    """Build the BC-ResNet-1 variant for NepSpot."""

    inputs = tf.keras.Input(shape=input_shape, name='mfcc_input')

    # Stem
    x = layers.Conv2D(
        16,
        kernel_size=(5, 5),
        strides=(2, 1),
        padding='same',
        use_bias=False,
        name='stem_conv',
    )(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_relu')(x)

    # Stage 1: 16 -> 8 channels
    x = _transition_block(x, 8, name='stage1_transition')
    x = _normal_block(x, 8, dilation_rate=1, name='stage1_block1')
    x = _normal_block(x, 8, dilation_rate=1, name='stage1_block2')

    # Stage 2: 8 -> 12 channels
    x = _transition_block(x, 12, name='stage2_transition')
    x = _normal_block(x, 12, dilation_rate=2, name='stage2_block1')
    x = _normal_block(x, 12, dilation_rate=2, name='stage2_block2')

    # Stage 3: 12 -> 16 channels
    x = _transition_block(x, 16, name='stage3_transition')
    x = _normal_block(x, 16, dilation_rate=4, name='stage3_block1')
    x = _normal_block(x, 16, dilation_rate=4, name='stage3_block2')
    x = _normal_block(x, 16, dilation_rate=4, name='stage3_block3')
    x = _normal_block(x, 16, dilation_rate=4, name='stage3_block4')

    # Stage 4: 16 -> 20 channels
    x = _transition_block(x, 20, name='stage4_transition')
    x = _normal_block(x, 20, dilation_rate=8, name='stage4_block1')
    x = _normal_block(x, 20, dilation_rate=8, name='stage4_block2')
    x = _normal_block(x, 20, dilation_rate=8, name='stage4_block3')
    x = _normal_block(x, 20, dilation_rate=8, name='stage4_block4')

    # Classifier head
    x = layers.Conv2D(
        32,
        kernel_size=(1, 5),
        padding='same',
        use_bias=False,
        name='head_conv1',
    )(x)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.ReLU(name='head_relu1')(x)

    x = layers.Conv2D(
        32,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name='head_conv2',
    )(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.ReLU(name='head_relu2')(x)

    x = layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    model = models.Model(inputs, outputs, name='BC_ResNet_NepSpot')
    return model


if __name__ == '__main__':
    model = build_bc_resnet()
    model.summary()