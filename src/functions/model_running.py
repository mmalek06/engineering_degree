import os
import shutil

import tensorflow as tf

from tensorflow import keras
from typing import Callable
from pathlib import Path

from functions.plotting import plot_single_output_history, plot_multi_output_history
from functions.loading_data import load_dataset, prepare_train_dataset, prepare_valid_dataset


def fit_model(
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        model_factory: Callable,
        checkpoint_path: str,
        log_path: str,
        reduction_patience: int = 5,
        monitor: str = 'val_accuracy',
        mode: str = 'max',
        stopping_patience: int = 10,
        steps_per_epoch: int = None,
        epochs: int = None,
        class_weight: dict[int, float] = None):
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
        mode=mode,
        save_best_only=True)
    tensor_board = keras.callbacks.TensorBoard(log_dir=log_path)
    model = model_factory()

    return model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight,
        callbacks=[reduce_lr, model_checkpoint, tensor_board, early_stopping]), model


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


def preserve_best_runs(source_name: str, dest_name: str, preserve_num: int) -> None:
    source_dir = Path(source_name)
    dest_dir = Path(dest_name)
    all_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    sorted_folders = sorted(all_folders, key=lambda x: x.stat().st_ctime)

    for folder in sorted_folders[-preserve_num:]:
        try:
            shutil.copytree(folder, dest_dir / folder.name)
        except FileExistsError:
            pass

    for folder in sorted_folders:
        shutil.rmtree(folder)


def finalize_run(root: str, plot_name: str, model_name: str, dataset_name: str, history: any,
                 plot_mode: str = 'single') -> None:
    plots_path = os.path.join(root, 'plots', dataset_name)
    models_path = os.path.join(root, 'models', dataset_name)

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    plot_path = os.path.join(plots_path, plot_name)

    increment_run_number(model_name)
    preserve_best_runs(
        os.path.join(root, 'tmp_models'),
        models_path,
        1)

    if plot_mode == 'single':
        plot_single_output_history(history.history, to_file=plot_path)
    elif plot_mode == 'multiple':
        plot_multi_output_history(history, to_file=plot_path)


def run_model(
        root: str,
        height: int,
        width: int,
        data_dir: str,
        dataset_name: str,
        model_base_name: str,
        model_getter: Callable,
        augmentation_getter: Callable,
        batch_size: int = 32,
        stopping_patience: int = 20,
        train_dataset: tf.data.Dataset = None,
        steps_per_epoch: int = None,
        epochs: int = 100,
        class_weight: dict[int, float] = None) -> (keras.Model, any):
    if train_dataset is None:
        train_dataset = load_dataset(width, height, data_dir, 'training', batch_size)

    num_classes = len(train_dataset.class_names)
    valid_dataset = load_dataset(width, height, data_dir, 'validation', batch_size)
    run_number = get_run_number(model_base_name)
    data_augmentation = augmentation_getter()
    train_dataset = prepare_train_dataset(train_dataset, data_augmentation)
    valid_dataset = prepare_valid_dataset(valid_dataset)
    model_name = f'{model_base_name}_{run_number}'
    logs_path = os.path.join(root, 'tensor_logs', dataset_name)

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    history, model = fit_model(
        train_dataset,
        valid_dataset,
        model_getter(num_classes),
        os.path.join(root, 'tmp_models', model_name + '_{epoch}'),
        os.path.join(logs_path, model_name),
        monitor='val_loss',
        mode='min',
        reduction_patience=10,
        stopping_patience=stopping_patience,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        class_weight=class_weight)
    plot_name = f'{model_name}.pdf'

    finalize_run(root, plot_name, model_base_name, dataset_name, history)

    return model, history
