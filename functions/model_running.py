import tensorflow as tf

from tensorflow import keras
from typing import Callable


def run_model(
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        model_factory: Callable,
        checkpoint_path: str,
        log_path: str,
        reduction_patience: int = 5,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        stopping_patience: int = 10):
    MIN_DELTA = .001
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=stopping_patience,
        min_delta=MIN_DELTA)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.95,
        min_delta=MIN_DELTA,
        patience=reduction_patience,
        min_lr=0.0005,
        verbose=1)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True)
    tensor_board = keras.callbacks.TensorBoard(log_dir=log_path)
    model = model_factory()

    return model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=100,
        batch_size=64,
        callbacks=[reduce_lr, model_checkpoint, tensor_board, early_stopping])
