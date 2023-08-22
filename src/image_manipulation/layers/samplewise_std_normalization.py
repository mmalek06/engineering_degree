import numpy as np
import tensorflow as tf

from tensorflow import keras

from .floodfill_mask import generate_mask


class SamplewiseStdNormalization(keras.layers.Layer):
    """
    Standardizes each sample so that it's of a unit standard deviation.
    """
    def __init__(self, epsilon=1e-7, **kwargs):
        super(SamplewiseStdNormalization, self).__init__(**kwargs)

        self.epsilon = epsilon

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        inputs_np = inputs.numpy()
        masks = np.array([generate_mask(img) for img in inputs_np])
        masks = np.expand_dims(masks, axis=-1)
        masks = np.repeat(masks, inputs_np.shape[-1], axis=-1)
        masks_tf = tf.convert_to_tensor(masks, dtype=tf.float32)
        mean, variance = tf.nn.weighted_moments(inputs, axes=[1, 2], frequency_weights=masks_tf)
        # in the context of taking a square root self.epsilon doesn't matter
        # but the square root value matters two lines below this one - better to avoid division by zero
        sample_stds = tf.sqrt(variance + self.epsilon)
        sample_stds = tf.reshape(sample_stds, [-1, 1, 1, inputs.shape[-1]])
        normalized_inputs = tf.where(sample_stds > self.epsilon,
                                     inputs / sample_stds,
                                     inputs)

        return normalized_inputs
