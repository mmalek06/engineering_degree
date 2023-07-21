import os
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL.Image

from tensorflow import keras

from typing import Tuple

SMALLER_WIDTH = 600 // 3
SMALLER_HEIGHT = 450 // 3


def parse_csv(file_path: str) -> Tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(file_path)
    filenames = df['image_id'].apply(lambda v: f'{v}.jpg').values
    x1 = df['left'].values.astype(np.float32) / SMALLER_WIDTH
    y1 = df['top'].values.astype(np.float32) / SMALLER_HEIGHT
    x2 = df['right'].values.astype(np.float32) / SMALLER_WIDTH
    y2 = df['bottom'].values.astype(np.float32) / SMALLER_HEIGHT

    return filenames, np.array([y1, x1, y2, x2]).T


def process_path(
        image_path: str,
        coords: np.ndarray,
        dot_targets: np.ndarray) -> Tuple[tf.Tensor, np.ndarray, np.ndarray]:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img, coords, dot_targets


def load_and_preprocess_data(csv_file_path: str, img_dir: str) -> tf.data.Dataset:
    image_filenames, coordinates = parse_csv(csv_file_path)
    image_filenames = [os.path.join(img_dir, fname) for fname in image_filenames]

    return tf.data.Dataset \
        .from_tensor_slices((
            image_filenames,
            coordinates,
            np.zeros((coordinates.shape[0],)))) \
        .map(process_path) \
        .batch(64) \
        .map(lambda img, coords, dots: (img, {'root': coords, 'dot': dots})) \
        .shuffle(1024)


def get_images_array(paths: list[str]) -> np.ndarray:
    rows = []

    for path in paths:
        with PIL.Image.open(path) as image:
            rescaled_image = np.asarray(image) / 255.

            rows.append(rescaled_image)

    return np.array(rows)


def get_name(path: str) -> str:
    return '_'.join(
        path
        .split(os.sep)[-1]
        .split('.')[-2])
