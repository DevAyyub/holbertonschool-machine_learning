#!/usr/bin/env python3
"""
Inception Network module for Deep CNNs
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
    Going Deeper with Convolutions (2014)

    Returns:
        The compiled keras model
    """
    X_input = K.Input(shape=(224, 224, 3))

    # Part 1: Initial Convolutions and Pooling
    X = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        activation='relu')(X_input)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    X = K.layers.Conv2D(filters=64,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        activation='relu')(X)
    X = K.layers.Conv2D(filters=192,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Part 2: Inception Blocks
    # Stage 3
    X = inception_block(X, [64, 96, 128, 16, 32, 32])       # 3a
    X = inception_block(X, [128, 128, 192, 32, 96, 64])     # 3b
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Stage 4
    X = inception_block(X, [192, 96, 208, 16, 48, 64])      # 4a
    X = inception_block(X, [160, 112, 224, 24, 64, 64])     # 4b
    X = inception_block(X, [128, 128, 256, 24, 64, 64])     # 4c
    X = inception_block(X, [112, 144, 288, 32, 64, 64])     # 4d
    X = inception_block(X, [256, 160, 320, 32, 128, 128])   # 4e
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Stage 5
    X = inception_block(X, [256, 160, 320, 32, 128, 128])   # 5a
    X = inception_block(X, [384, 192, 384, 48, 128, 128])   # 5b

    # Part 3: Classifier Output
    # Average pooling squashes the 7x7 spatial dimensions to 1x1
    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(1, 1))(X)

    # Dropout layer (GoogLeNet specifies 40% dropout)
    X = K.layers.Dropout(rate=0.4)(X)

    # Output layer
    X = K.layers.Dense(units=1000, activation='softmax')(X)

    # Create the model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
