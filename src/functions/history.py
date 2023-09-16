import os

import numpy as np


def save_history(history, name: str, root_path: str = None) -> None:
    if root_path is None:
        root_path = os.path.join('..', '..', '..', '..')

    np.save(
        os.path.join(
            root_path,
            'histories',
            f'{name}.npy'), history.history)


def load_history(name: str, root_path: str = None):
    if root_path is None:
        root_path = os.path.join('..', '..', '..', '..')

    return np.load(
        os.path.join(
            root_path,
            'histories',
            f'{name}.npy'),
        allow_pickle=True).item()
