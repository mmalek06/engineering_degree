import numpy as np
import tensorflow as tf

from tensorflow import keras

from .floodfill_mask import generate_mask


class SamplewiseCenter(keras.layers.Layer):
    """
    This class centers each individual image around zero. It can help if there's
    a lot of variation in brightness or color scheme.
    """
    def __init__(self, **kwargs):
        super(SamplewiseCenter, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        batch_masks = np.array([generate_mask(img) for img in inputs.numpy()])
        batch_masks = tf.convert_to_tensor(batch_masks, dtype=tf.float32)
        batch_masks = tf.expand_dims(batch_masks, axis=-1)
        included_pixel_sum = tf.reduce_sum(inputs * batch_masks, axis=[1, 2], keepdims=True)
        count = tf.reduce_sum(batch_masks, axis=[1, 2], keepdims=True) + 1e-7
        sample_means = included_pixel_sum / count
        centered_inputs = inputs - sample_means

        return centered_inputs
