#!/usr/bin/env python3
"""Module to create a convolutional autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (H, W, C).
        filters (list): Number of filters for each convolutional layer in the
            encoder, respectively.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs_encoder = keras.Input(shape=input_dims)
    x = inputs_encoder
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=inputs_encoder, outputs=x)

    # Decoder
    inputs_decoder = keras.Input(shape=latent_dims)
    x = inputs_decoder
    rev_filters = list(reversed(filters))

    # All decoder conv layers except second to last and last
    for i in range(len(rev_filters) - 1):
        x = keras.layers.Conv2D(rev_filters[i], (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Second to last convolution
    x = keras.layers.Conv2D(rev_filters[-1], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Last convolution
    outputs_decoder = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                          activation='sigmoid',
                                          padding='same')(x)
    decoder = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

    # Autoencoder
    auto_outputs = decoder(encoder(inputs_encoder))
    auto = keras.Model(inputs=inputs_encoder, outputs=auto_outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
