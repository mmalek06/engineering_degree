import tensorflow as tf

from tensorflow import keras


class FeaturewiseStdNormalization(keras.layers.Layer):
    """
    Standardizes the input so that it's of unit standard deviation.
    """
    def __init__(self, **kwargs):
        super(FeaturewiseStdNormalization, self).__init__(**kwargs)

        self._std = tf.zeros((1, 1))

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        if self._std is None:
            raise ValueError('The layer needs to be adapted before usage.')

        return inputs / self._std

    def adapt(self, data: tf.raw_ops.BatchDataset):
        self.build(data.element_spec[0].shape)

        pixel_sum = tf.zeros([1, 1, 3])
        pixel_squared_sum = tf.zeros([1, 1, 3])
        num_samples = 0

        for img_batch, _ in data:
            current_batch_size = img_batch.shape[0]
            pixel_sum += tf.reduce_sum(img_batch, axis=[0, 1, 2])
            pixel_squared_sum += tf.reduce_sum(tf.square(img_batch), axis=[0, 1, 2])
            num_samples += current_batch_size * tf.reduce_prod(img_batch.shape[1:3])

        mean = pixel_sum / tf.cast(num_samples, tf.float32)
        variance = (pixel_squared_sum / tf.cast(num_samples, tf.float32)) - tf.square(mean)
        self._std = tf.sqrt(variance)
