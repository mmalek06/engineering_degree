import tensorflow as tf

from tensorflow import keras


class ZCAWhitening(keras.layers.Layer):
    """
    This class centers the dataset around 0 and also decorelates features.
    """

    _means: tf.Tensor
    _covariance: tf.Tensor

    def __init__(self, epsilon: float = 1e-10, **kwargs):
        super(ZCAWhitening, self).__init__(**kwargs)

        self._epsilon = epsilon
        self._whitening_matrix = None

    def adapt(self, data: tf.raw_ops.BatchDataset):
        self.build(data.element_spec[0].shape)

        self._means = tf.zeros(shape=[data.element_spec[0].shape[-1]])
        self._covariance = tf.zeros(shape=[data.element_spec[0].shape[-1], data.element_spec[0].shape[-1]])
        count = tf.constant(0.0)

        for x, _ in data:
            x = tf.reshape(x, (x.shape[0], -1))
            batch_mean = tf.reduce_mean(x, axis=0)
            batch_cov = tf.matmul(tf.transpose(x - batch_mean), x - batch_mean) / tf.cast(tf.shape(x)[0], tf.float32)
            self._means += tf.reduce_sum(x, axis=0)
            self._covariance += batch_cov * tf.cast(tf.shape(x)[0], tf.float32)
            count += tf.cast(tf.shape(x)[0], tf.float32)

        self._means /= count
        self._covariance /= count
        s, u, _ = tf.linalg.svd(self._covariance)
        s_inv = tf.linalg.diag(1.0 / tf.sqrt(s + self._epsilon))
        self._whitening_matrix = tf.matmul(tf.matmul(u, s_inv), u, transpose_b=True)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        if self._whitening_matrix is None:
            raise ValueError('The layer needs to be adapted before usage.')

        original_shape = tf.shape(inputs)
        flat_shape = (-1, original_shape[-3] * original_shape[-2], original_shape[-1])
        x = tf.reshape(inputs, flat_shape)
        x_centered = x - self._means
        x_whitened = tf.linalg.matmul(x_centered, self._whitening_matrix)

        return tf.reshape(x_whitened, original_shape)
