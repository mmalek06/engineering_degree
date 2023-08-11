import os
import shutil

import tensorflow as tf

from tensorflow import keras
from typing import Callable
from pathlib import Path


def run_model(
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        model_factory: Callable,
        checkpoint_path: str,
        log_path: str,
        reduction_patience: int = 5,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        stopping_patience: int = 15):
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


def _get_runs_file_path(run_file: str) -> str:
    return os.path\
        .join(
            os.path.expanduser('~'),
            f'.{run_file}')


def get_run_number(run_file: str) -> int:
    run_files_path = _get_runs_file_path(run_file)
    run_file = Path(run_files_path)

    run_file.parent.mkdir(exist_ok=True, parents=True)
    run_file.touch(exist_ok=True)

    text = run_file.read_text()

    if len(text) > 0:
        return int(text)

    return 1


def increment_run_number(run_file: str) -> None:
    number = str(get_run_number(run_file) + 1)
    run_files_path = _get_runs_file_path(run_file)
    file_path = Path(run_files_path)

    file_path.write_text(number)


def preserve_best_runs(source_name: str, dest_name: str) -> None:
    source_dir = Path(source_name)
    dest_dir = Path(dest_name)
    all_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    sorted_folders = sorted(all_folders, key=lambda x: x.stat().st_ctime)

    for folder in sorted_folders[-3:]:
        try:
            shutil.copytree(folder, dest_dir / folder.name)
        except FileExistsError:
            pass

    for folder in sorted_folders:
        shutil.rmtree(folder)
