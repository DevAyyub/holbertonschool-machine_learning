#!/usr/bin/env python3
"""
Module that creates a variational autoencoder (VAE) using tensorflow.keras.
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in
            the encoder, respectively.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: the encoder model outputting [latent, mean, log_var]
            - decoder: the decoder model
            - auto: the full autoencoder model
    """
    K = keras.backend

    # 1. Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(units=latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(units=latent_dims, activation=None)(x)

    def sampling(args):
        """Samples from the latent space using the reparameterization trick."""
        mu, log_var = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return mu + K.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=[z, z_mean, z_log_var],
        name='encoder'
    )

    # 2. Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    y = decoder_inputs
    for nodes in reversed(hidden_layers):
        y = keras.layers.Dense(units=nodes, activation='relu')(y)

    decoder_outputs = keras.layers.Dense(
        units=input_dims,
        activation='sigmoid'
    )(y)

    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        name='decoder'
    )

    # 3. Full Variational Autoencoder
    outputs = decoder(encoder(encoder_inputs)[0])
    auto = keras.Model(
        inputs=encoder_inputs,
        outputs=outputs,
        name='autoencoder'
    )

    # VAE Loss (Reconstruction + KL Divergence)
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs, outputs
    )
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
