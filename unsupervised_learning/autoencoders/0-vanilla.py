#!/usr/bin/env python3
"""Module to create a vanilla autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder.

    Args:
        input_dims (int): Dimension of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in encoder.
        latent_dims (int): Dimension of the latent space representation.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs_encoder = keras.Input(shape=(input_dims,))
    x = inputs_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=inputs_encoder, outputs=latent)

    # Decoder
    inputs_decoder = keras.Input(shape=(latent_dims,))
    x = inputs_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs_decoder = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

    # Autoencoder
    auto_outputs = decoder(encoder(inputs_encoder))
    auto = keras.Model(inputs=inputs_encoder, outputs=auto_outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
