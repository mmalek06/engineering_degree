import os
import numpy as np

from tensorflow import keras


def _count(path: str) -> int:
    count = 0

    for path in os.scandir(path):
        if path.is_file():
            count += 1

    return count


def calculate_initial_biases(root_path: str) -> keras.initializers.Constant:
    categories = os.listdir(root_path)
    counts = []

    for category in categories:
        category_path = os.path.join(root_path, category)
        count = _count(category_path)

        counts.append(count)

    total_samples = sum(counts)
    log_odds = np.log([count/total_samples / (1 - count/total_samples) for count in counts])

    return keras.initializers.Constant(log_odds)
