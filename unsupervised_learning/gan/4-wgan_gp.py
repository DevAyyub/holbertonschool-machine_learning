#!/usr/bin/env python3
"""
WGAN_GP module with weight replacement support.
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    WGAN_GP model class inheriting from keras.Model.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
        lambda_gp=10,
    ):
        """
        Initializes the WGAN_GP model.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3
        self.beta_2 = 0.9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, int(self.len_dims)):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Define generator loss and optimizer
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # Define discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y: tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a fake sample batch from the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Extracts a random batch from real examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates an interpolated sample between real and fake samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the discriminator.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """
        Performs one training step for discriminator and generator.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample()
                interpolated_samples = self.get_interpolated_sample(
                    real_samples, fake_samples
                )
                discr_loss = self.discriminator.loss(
                    self.discriminator(fake_samples),
                    self.discriminator(real_samples),
                )
                gp = self.gradient_penalty(interpolated_samples)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            discr_gradients = tape.gradient(
                new_discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample()
            fake_output = self.discriminator(fake_samples)
            gen_loss = self.generator.loss(fake_output)
        gen_gradients = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replaces generator and discriminator weights from .h5 files.

        Args:
            gen_h5: Path to generator weights .h5 file.
            disc_h5: Path to discriminator weights .h5 file.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def replace_weight(self, gen_h5, disc_h5):
        """
        Alias for replace_weights.
        """
        self.replace_weights(gen_h5, disc_h5)
