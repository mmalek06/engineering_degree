import tensorflow as tf

from tensorflow import keras


class FeaturewiseCenter(keras.layers.Layer):
    """
    This class centers the whole dataset around 0. If there's a dominating color scheme
    or background then normalizing images feature-wise can help model converge better.
    """
    def __init__(self, **kwargs):
        super(FeaturewiseCenter, self).__init__(**kwargs)

        self._mean = tf.zeros((1, 1))

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        if self._mean is None:
            raise ValueError('The layer needs to be adapted before usage.')

        return inputs - self._mean

    def adapt(self, data: tf.raw_ops.BatchDataset):
        self.build(data.element_spec[0].shape)

        pixel_sum = tf.zeros([1, 1, 3])
        num_samples = 0

        for img_batch, _ in data:
            current_batch_size = img_batch.shape[0]
            pixel_sum += tf.reduce_sum(img_batch, axis=[0, 1, 2])
            num_samples += current_batch_size * tf.reduce_prod(img_batch.shape[1:3])

        self._mean = pixel_sum / tf.cast(num_samples, tf.float32)
