#!/usr/bin/env python3
"""Module to create a sparse autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha=10e-6):
    """Creates a sparse autoencoder.

    Args:
        input_dims (int): Dimension of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in encoder.
        latent_dims (int): Dimension of the latent space representation.
        lambtha (float, optional): Regularization parameter used for L1
            regularization on the encoded output. Defaults to 10e-6.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The sparse autoencoder model.
    """
    if isinstance(input_dims, (tuple, list)):
        input_shape = tuple(input_dims)
        in_dim = input_dims[0]
    else:
        input_shape = (input_dims,)
        in_dim = input_dims

    if isinstance(latent_dims, (tuple, list)):
        latent_shape = tuple(latent_dims)
        lat_dim = latent_dims[0]
    else:
        latent_shape = (latent_dims,)
        lat_dim = latent_dims

    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]

    # Encoder
    inputs_encoder = keras.Input(shape=input_shape)
    x = inputs_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    if lambtha is not None:
        reg = keras.regularizers.l1(lambtha)
    else:
        reg = None

    latent = keras.layers.Dense(
        lat_dim,
        activation='relu',
        activity_regularizer=reg
    )(x)
    encoder = keras.Model(inputs=inputs_encoder, outputs=latent)

    # Decoder
    inputs_decoder = keras.Input(shape=latent_shape)
    y = inputs_decoder
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(nodes, activation='relu')(y)
    outputs_decoder = keras.layers.Dense(in_dim, activation='sigmoid')(y)
    decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

    # Autoencoder
    auto_outputs = decoder(encoder(inputs_encoder))
    auto = keras.Model(inputs=inputs_encoder, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
