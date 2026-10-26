#!/usr/bin/env python3
"""Module to create a variational autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims (int): Dimension of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in encoder.
        latent_dims (int): Dimension of the latent space representation.

    Returns:
        encoder: Encoder model outputting [latent, mean, log_variance].
        decoder: Decoder model.
        auto: Full variational autoencoder model.
    """
    # ------------------ ENCODER ------------------
    inputs_encoder = keras.Input(shape=(input_dims,))
    x = inputs_encoder
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Samples latent points using reparameterization trick."""
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    latent = keras.layers.Lambda(sampling)([mu, log_sig])
    encoder = keras.Model(inputs=inputs_encoder,
                          outputs=[latent, mu, log_sig])

    # ------------------ DECODER ------------------
    inputs_decoder = keras.Input(shape=(latent_dims,))
    x = inputs_decoder
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs_decoder = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

    # ---------------- FULL AUTOENCODER -----------
    encoded_latent, enc_mu, enc_log_sig = encoder(inputs_encoder)
    auto_outputs = decoder(encoded_latent)
    auto = keras.Model(inputs=inputs_encoder, outputs=auto_outputs)

    def vae_loss(y_true, y_pred):
        """Computes combined reconstruction and KL divergence loss."""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true, y_pred) * input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + enc_log_sig - keras.backend.square(enc_mu) -
            keras.backend.exp(enc_log_sig),
            axis=-1
        )
        return keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
