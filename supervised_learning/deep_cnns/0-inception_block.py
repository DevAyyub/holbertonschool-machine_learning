#!/usr/bin/env python3
"""
Inception Block module for Deep CNNs
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014).

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F1, F3R, F3, F5R, F5, FPP:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 conv before 3x3 conv
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 conv before 5x5 conv
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 conv after max pooling

    Returns:
        The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Path 1: 1x1 Convolution
    conv_1x1 = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)

    # Path 2: 1x1 Convolution -> 3x3 Convolution
    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)
    conv_3x3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), padding='same', activation='relu'
    )(conv_3x3_reduce)

    # Path 3: 1x1 Convolution -> 5x5 Convolution
    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1), padding='same', activation='relu'
    )(A_prev)
    conv_5x5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5), padding='same', activation='relu'
    )(conv_5x5_reduce)

    # Path 4: 3x3 Max Pooling -> 1x1 Convolution
    pool_proj = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same'
    )(A_prev)
    conv_pool = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1), padding='same', activation='relu'
    )(pool_proj)

    # Concatenate all paths together along the channel axis
    output = K.layers.Concatenate(axis=3)(
        [conv_1x1, conv_3x3, conv_5x5, conv_pool]
    )

    return output
