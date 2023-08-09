import tensorflow as tf

from tensorflow import keras


class SamplewiseStdNormalization(keras.layers.Layer):
    """
    Standardizes each sample so that it's of a unit standard deviation.
    """
    def __init__(self, **kwargs):
        super(SamplewiseStdNormalization, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        sample_stds = tf.math.reduce_std(inputs, axis=[1, 2], keepdims=True)
        normalized_inputs = inputs / sample_stds

        return normalized_inputs
