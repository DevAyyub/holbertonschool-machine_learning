#!/usr/bin/env python3
"""Module for creating masks for Transformer"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in)
        target: tf.Tensor of shape (batch_size, seq_len_out)

    Returns:
        encoder_mask: padding mask of shape (batch_size, 1, 1, seq_len_in)
        combined_mask: shape (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: padding mask of shape (batch_size, 1, 1, seq_len_in)
    """
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask (same as the encoder padding mask)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Look ahead mask
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Decoder target padding mask
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis,
                                                      tf.newaxis, :]

    # Combined mask (look ahead combined with decoder target padding mask)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
