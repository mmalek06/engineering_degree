import os

import numpy as np


def save_history(history, name: str) -> None:
    np.save(
        os.path.join(
            '..',
            '..',
            'histories',
            f'{name}.npy'), history.history)


def load_history(name: str):
    return np.load(
        os.path.join(
            '..',
            '..',
            'histories',
            f'{name}.npy'),
        allow_pickle=True).item()
