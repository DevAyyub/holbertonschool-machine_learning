#!/usr/bin/env python3
"""
Module to save and load Keras models
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file

    Args:
        network: the model to save
        filename: the path of the file that the model should be saved to

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file

    Args:
        filename: the path of the file that the model should be loaded from

    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
