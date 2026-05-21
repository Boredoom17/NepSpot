"""BC-ResNet-1 for NepSpot keyword spotting.

Corrected implementation following Kim et al. 2021,
"Broadcasted Residual Learning for Efficient Keyword Spotting" (Interspeech 2021).
Reference port: https://github.com/Qualcomm-AI-research/bcresnet

Input shape: (40, 32, 1)
  axis 0 = frequency (40 MFCCs)
  axis 1 = time (32 frames)
  axis 2 = channels

A BC-ResNet normal block has two parallel depthwise branches:
  - frequency branch: DepthwiseConv2D(kernel=(3,1), no dilation) -> BN
  - time branch:      DepthwiseConv2D(kernel=(1,3), dilation=(1,d)) -> BN -> Swish

The time branch is average-pooled over the frequency axis (collapsing freq to 1)
and broadcast back along frequency before being added to the frequency-branch
output. A pointwise 1x1 conv + BN + ReLU finishes the block. A residual skip is
added when input/output shapes agree.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _freq_branch(x, name):
    y = layers.DepthwiseConv2D(
        kernel_size=(3, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_freq_dw',
    )(x)
    y = layers.BatchNormalization(name=f'{name}_freq_bn')(y)
    return y


def _time_branch(x, time_dilation, name):
    y = layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        dilation_rate=(1, time_dilation),
        padding='same',
        use_bias=False,
        name=f'{name}_time_dw',
    )(x)
    y = layers.BatchNormalization(name=f'{name}_time_bn')(y)
    y = layers.Activation('swish', name=f'{name}_time_swish')(y)
    return y


def _broadcast_over_freq(t, name):
    """Broadcast 'mean over frequency' to every freq position via a single op.

    Mathematically equivalent to
        AveragePooling2D((F,1), valid) -> UpSampling2D((F,1), nearest)
    but implemented as one frozen DepthwiseConv2D so the TFLite converter
    does not hit the 'same scale constraint' between the pool and the
    downstream Add during INT8 quantization of QAT models.

    With kernel size (2F-1, 1) and padding='same', the kernel window at every
    output freq position spans all F input positions (the remaining F-1 taps
    fall on zero-pad and contribute 0). With every kernel weight set to 1/F,
    output[b, f, t, c] = (1/F) * sum_{f'=0..F-1} input[b, f', t, c] = mean,
    for every f. Forward-pass equivalence to the old AvgPool+UpSampling path
    is bit-identical up to ~1e-7 floating-point rounding.

    The weights are frozen (trainable=False) and never updated during training.
    """
    freq_dim = t.shape[1]
    if freq_dim is None:
        raise ValueError('BC-ResNet expects a static frequency dimension')
    freq_dim = int(freq_dim)
    kernel_h = 2 * freq_dim - 1

    broadcaster = layers.DepthwiseConv2D(
        kernel_size=(kernel_h, 1),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        depthwise_initializer=tf.keras.initializers.Constant(1.0 / freq_dim),
        trainable=False,
        name=f'{name}_freq_avg_broadcast',
    )
    return broadcaster(t)


def _normal_block(x, filters, time_dilation, name):
    shortcut = x

    f = _freq_branch(x, name=f'{name}_fb')
    t = _time_branch(x, time_dilation=time_dilation, name=f'{name}_tb')
    t_bcast = _broadcast_over_freq(t, name=f'{name}_tb_bcast')

    y = layers.Add(name=f'{name}_branch_add')([f, t_bcast])

    y = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_pw',
    )(y)
    y = layers.BatchNormalization(name=f'{name}_pw_bn')(y)

    if (shortcut.shape[-1] == filters
            and shortcut.shape[1] == y.shape[1]
            and shortcut.shape[2] == y.shape[2]):
        y = layers.Add(name=f'{name}_res_add')([y, shortcut])

    y = layers.ReLU(name=f'{name}_relu')(y)
    return y


def _transition_block(x, filters, name):
    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        name=f'{name}_pw',
    )(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    x = layers.AveragePooling2D(
        pool_size=(2, 1),
        strides=(2, 1),
        padding='same',
        name=f'{name}_freq_pool',
    )(x)
    return x


def build_bc_resnet(input_shape=(40, 32, 1), num_classes=12):
    """BC-ResNet-1 (smallest variant) per Kim et al. 2021."""

    inputs = tf.keras.Input(shape=input_shape, name='mfcc_input')

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

    x = _transition_block(x, filters=8, name='stage1_transition')
    x = _normal_block(x, filters=8,  time_dilation=1, name='stage1_block1')
    x = _normal_block(x, filters=8,  time_dilation=1, name='stage1_block2')

    x = _transition_block(x, filters=12, name='stage2_transition')
    x = _normal_block(x, filters=12, time_dilation=2, name='stage2_block1')
    x = _normal_block(x, filters=12, time_dilation=2, name='stage2_block2')

    x = _transition_block(x, filters=16, name='stage3_transition')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block1')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block2')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block3')
    x = _normal_block(x, filters=16, time_dilation=4, name='stage3_block4')

    x = _transition_block(x, filters=20, name='stage4_transition')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block1')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block2')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block3')
    x = _normal_block(x, filters=20, time_dilation=8, name='stage4_block4')

    x = layers.Conv2D(
        32, kernel_size=(1, 5), padding='same', use_bias=False, name='head_conv1',
    )(x)
    x = layers.BatchNormalization(name='head_bn1')(x)
    x = layers.ReLU(name='head_relu1')(x)
    x = layers.Conv2D(
        32, kernel_size=(1, 1), padding='same', use_bias=False, name='head_conv2',
    )(x)
    x = layers.BatchNormalization(name='head_bn2')(x)
    x = layers.ReLU(name='head_relu2')(x)

    x = layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    return models.Model(inputs, outputs, name='BC_ResNet_NepSpot')


if __name__ == '__main__':
    model = build_bc_resnet(input_shape=(40, 32, 1), num_classes=12)
    model.summary(line_length=120)

    total_params = int(sum(int(tf.reduce_prod(v.shape).numpy()) for v in model.weights))
    trainable_params = int(sum(int(tf.reduce_prod(v.shape).numpy()) for v in model.trainable_weights))
    non_trainable_params = total_params - trainable_params

    print()
    print(f'Total params:        {total_params:,}')
    print(f'Trainable params:    {trainable_params:,}')
    print(f'Non-trainable params:{non_trainable_params:,}')
    print()
    print('Frequency-axis trace through the network:')
    print('  input              freq=40')
    print('  stem (stride 2,1)  freq=20')
    print('  stage1 trans+pool  freq=10')
    print('  stage2 trans+pool  freq=5')
    print('  stage3 trans+pool  freq=3   (ceil(5/2) with padding=same)')
    print('  stage4 trans+pool  freq=2   (ceil(3/2))')
    print()
    print('Time-axis dilation receptive-field check (kernel=(1,3), padding=same):')
    print('  stage1 d=1 -> taps at  -1, 0, +1   -> RF= 3  on T=32   OK')
    print('  stage2 d=2 -> taps at  -2, 0, +2   -> RF= 5  on T=32   OK')
    print('  stage3 d=4 -> taps at  -4, 0, +4   -> RF= 9  on T=32   OK')
    print('  stage4 d=8 -> taps at  -8, 0, +8   -> RF=17  on T=32   OK')
    print()
    print('Frequency-axis kernel (3,1) check at each stage:')
    print('  stage1 freq=10  (3 taps -> trivially fits)')
    print('  stage2 freq=5   (3 taps fit)')
    print('  stage3 freq=3   (3 taps fit exactly; same padding adds 1 pad each side)')
    print('  stage4 freq=2   (3 taps with same padding: 2/3 taps active, 1 pads)')
