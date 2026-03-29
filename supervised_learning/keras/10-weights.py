#!/usr/bin/env python3
"""
Module to save and load Keras model weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file

    Args:
        network: the model whose weights should be saved
        filename: the path of the file to save the weights to
        save_format: the format in which the weights should be saved

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights from a file

    Args:
        network: the model to which the weights should be loaded
        filename: the path of the file to load the weights from

    Returns:
        None
    """
    network.load_weights(filename)
