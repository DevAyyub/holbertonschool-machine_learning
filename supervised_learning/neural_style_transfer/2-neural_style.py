#!/usr/bin/env python3
"""
Neural Style Transfer module
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for neural style transfer.
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for NST.
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels.
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        max_dim = 512
        h, w, _ = image.shape
        scale = max_dim / max(h, w)
        h_new = int(h * scale)
        w_new = int(w * scale)

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, [h_new, w_new], method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        x = vgg.input
        outputs = {}

        # Reconstruct the model up to block5_conv2,
        # replacing MaxPool with AvgPool
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    name=layer.name)(x)
            else:
                x = layer(x)

            outputs[layer.name] = x

            # Break early once we hit the content layer
            if layer.name == self.content_layer:
                break

        output_layers = self.style_layers + [self.content_layer]
        model_outputs = [outputs[name] for name in output_layers]

        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # einsum computes dot product across height and width (i and j)
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Get spatial dimensions (H and W) to compute the denominator
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return result / num_locations
