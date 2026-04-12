#!/usr/bin/env python3
"""
Module that contains the function pca_color
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: A 3D tf.Tensor containing the image to change.
        alphas: A tuple of length 3 containing the amount that each channel
                should change.

    Returns:
        The augmented image.
    """
    img = tf.cast(image, tf.float32)
    flat = tf.reshape(img, (-1, 3))
    
    # Mean center the pixels
    mean = tf.reduce_mean(flat, axis=0)
    centered = flat - mean
    
    # Compute the covariance matrix
    n = tf.cast(tf.shape(centered)[0] - 1, tf.float32)
    cov = tf.tensordot(tf.transpose(centered), centered, axes=1) / n
    
    # Eigen decomposition (PCA)
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)
    
    # Calculate the perturbation (delta)
    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)
    delta = tf.reduce_sum(eigenvectors * (eigenvalues * alphas), axis=1)
    
    # Apply the perturbation to the original image
    pca_image = img + delta
    
    # Clip values to valid image range and cast back to original type
    pca_image = tf.clip_by_value(pca_image, 0.0, 255.0)
    return tf.cast(pca_image, image.dtype)
