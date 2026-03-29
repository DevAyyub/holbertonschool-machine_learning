#!/usr/bin/env python3
"""
Module to save and load Keras model configurations in JSON format
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

    Args:
        network: the model whose configuration should be saved
        filename: the path of the file to save the configuration to

    Returns:
        None
    """
    # Convert model architecture to JSON string
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration

    Args:
        filename: the path of the file containing the JSON configuration

    Returns:
        The loaded model
    """
    with open(filename, "r") as f:
        config = f.read()
    # Reconstruct the model from the JSON string
    return K.models.model_from_json(config)
