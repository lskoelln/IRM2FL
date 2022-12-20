import tensorflow as tf
from tensorflow.python.ops import math_ops

from keras.utils import losses_utils

MS_SSIM_POWER_FACTORS = [ [1.0],
                          [0.40678, 0.59323],
                          [0.2096, 0.4659, 0.3245],
                          [0.11063, 0.33253, 0.33513, 0.2217],
                          [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] ]


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def _normalize(img1, img2, max_val=1.0):
    maximum = tf.math.reduce_max([tf.math.reduce_max(img1), tf.math.reduce_max(img2)])
    minimum = tf.math.reduce_min([tf.math.reduce_min(img1), tf.math.reduce_min(img2)])
    return map(lambda x: (x - minimum) / (maximum - minimum) * max_val, (img1, img2))


class PSNR(tf.keras.metrics.Mean):
    def __init__(self, name='PSNR', dtype=None, max_val=1.0, normalize=False):
        super(PSNR, self).__init__(name, dtype=dtype)
        self.max_val = max_val
        self.normalize = normalize

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_true, y_pred)
        if self.normalize: y_true, y_pred = _normalize(y_true, y_pred, max_val=self.max_val)
        mse = math_ops.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        output = 10 * log10(math_ops.div_no_nan(math_ops.square(self.max_val), mse))
        return super(PSNR, self).update_state(output, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class NRMSE(tf.keras.metrics.Mean):
    """
    Largely copied from Tensorflow v2.5.0 > tf.keras.metrics.RootMeanSquaredError
    Returns same values as 'tf.keras.metrics.RootMeanSquaredError' if data_range=1.0.
    """
    def __init__(self, name='NRMSE', dtype=None, data_range=1.0):
        super(NRMSE, self).__init__(name, dtype=dtype)
        self.data_range = data_range

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        error_sq = math_ops.squared_difference(y_pred, y_true)
        return super(NRMSE, self).update_state(error_sq, sample_weight=sample_weight)

    def result(self):
        return math_ops.divide(math_ops.sqrt(math_ops.div_no_nan(self.total, self.count)), self.data_range)


class MS_SSIM(tf.keras.metrics.Mean):
    def __init__(self, name='MS_SSIM', dtype=None, max_val=1.0, normalize=False, crop=False,
                 n_scales=1, filter_size=11):
        super(MS_SSIM, self).__init__(name, dtype=dtype)
        self.max_val = max_val
        self.crop = crop
        self.normalize = normalize
        self.power_factors = MS_SSIM_POWER_FACTORS[n_scales-1]
        self.filter_size = filter_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_true, y_pred)
        if self.crop:
            y_pred = tf.clip_by_value(y_pred, 0, self.max_val)
            y_true = tf.clip_by_value(y_true, 0, self.max_val)
        if self.normalize: y_true, y_pred = _normalize(y_true, y_pred, max_val=self.max_val)
        output = tf.image.ssim_multiscale(y_pred, y_true, max_val=self.max_val, power_factors=self.power_factors, filter_size=self.filter_size)
        return super(MS_SSIM, self).update_state(output, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class MAE(tf.keras.metrics.Mean):
    def __init__(self, name='MAE', dtype=None):
        super(MAE, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        output = math_ops.reduce_mean(math_ops.abs(math_ops.subtract(y_pred, y_true)))
        return super(MAE, self).update_state(output, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)


class MSE(tf.keras.metrics.Mean):
    def __init__(self, name='MSE', dtype=None):
        super(MSE, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates root mean squared error statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

        output = math_ops.reduce_mean(math_ops.squared_difference(y_pred, y_true))
        return super(MSE, self).update_state(output, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)

    
class NCC(tf.keras.metrics.Mean):
    def __init__(self, name='NCC', dtype=None):
        super(NCC, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_true, y_pred)
    
        sigma_true = tf.math.reduce_std(y_true)
        sigma_pred = tf.math.reduce_std(y_pred)

        y_true = y_true - tf.math.reduce_mean(y_true)
        y_pred = y_pred - tf.math.reduce_mean(y_pred)

        output = math_ops.reduce_mean(y_true*y_pred/(sigma_true*sigma_pred))

        return super(NCC, self).update_state(output, sample_weight=sample_weight)

    def result(self):
        return math_ops.div_no_nan(self.total, self.count)
