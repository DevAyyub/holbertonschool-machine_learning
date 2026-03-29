#!/usr/bin/env python3
"""
Module to train a Keras model with validation data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and validation data

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with, if not None
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
