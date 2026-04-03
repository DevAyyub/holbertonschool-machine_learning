#!/usr/bin/env python3
"""
Module to build a modified LeNet-5 architecture using Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture using keras
    """
    init = K.initializers.HeNormal(seed=0)

    # Convolutional layer 1
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Max pooling layer 1
    pool1 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Convolutional layer 2
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # Max pooling layer 2
    pool2 = K.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten
    flatten = K.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # Fully connected layer 2
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # Output layer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
