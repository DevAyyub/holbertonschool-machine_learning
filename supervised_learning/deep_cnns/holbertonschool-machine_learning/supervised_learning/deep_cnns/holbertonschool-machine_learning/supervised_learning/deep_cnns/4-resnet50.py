#!/usr/bin/env python3
"""
ResNet-50 architecture module for Deep CNNs
"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).

    Returns:
        The compiled keras model
    """
    X_input = K.Input(shape=(224, 224, 3))
    init = K.initializers.HeNormal(seed=0)

    # Stage 1
    # 7x7 Conv, 64 filters, stride 2
    X = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    # 3x3 Max Pool, stride 2
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Stage 2
    # Note: First block uses s=1 because MaxPool already downsampled
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, [128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, [256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, [512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # Average Pooling (7x7 spatial dimensions -> 1x1)
    X = K.layers.AveragePooling2D(pool_size=(7, 7))(X)

    # Fully Connected Layer (Dense)
    X = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer=init)(X)

    # Create model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
