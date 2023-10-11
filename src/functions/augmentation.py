import tensorflow as tf

from tensorflow import keras

from image_manipulation.layers.samplewise_center import SamplewiseCenter


def _get_basic_layers() -> list[keras.layers.Layer]:
    return [
        keras.layers.RandomFlip('horizontal_and_vertical'),
        keras.layers.RandomRotation(1),
        keras.layers.RandomBrightness((-.3, .3)),
        keras.layers.RandomContrast(.3),
        keras.layers.RandomZoom((.3, -.3), (.3, -.3))
    ]


def get_augmentation_layers() -> keras.Sequential:
    return tf.keras.Sequential(_get_basic_layers())


def get_augmentation_layers_with_sample_augmentation() -> keras.Sequential:
    centering = SamplewiseCenter()
    basic_layers = _get_basic_layers()

    return tf.keras.Sequential([
        centering,
    ] + basic_layers)
