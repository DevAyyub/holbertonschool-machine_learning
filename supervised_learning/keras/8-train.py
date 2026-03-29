#!/usr/bin/env python3
"""
Module to train a Keras model with early stopping,
learning rate decay, and model checkpointing
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with multiple callbacks

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: data to validate the model with
        early_stopping: boolean indicating whether to use early stopping
        patience: patience used for early stopping
        learning_rate_decay: boolean indicating whether to use decay
        alpha: initial learning rate
        decay_rate: decay rate for inverse time decay
        save_best: boolean indicating whether to save the best model
        filepath: file path where the model should be saved
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches

    Returns:
        The History object generated after training the model
    """
    callbacks = []

    # 1. Early Stopping
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    # 2. Learning Rate Decay
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """Inverse time decay function"""
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

    # 3. Save Best Model
    if validation_data and save_best and filepath:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
