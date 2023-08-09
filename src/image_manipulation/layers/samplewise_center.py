import tensorflow as tf

from tensorflow import keras


class SamplewiseCenter(keras.layers.Layer):
    """
    This class centers each individual image around zero. It can help if there's
    a lot of variation in brightness or color scheme.
    """
    def __init__(self, **kwargs):
        super(SamplewiseCenter, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        sample_means = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        centered_inputs = inputs - sample_means

        return centered_inputs
