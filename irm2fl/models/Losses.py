import tensorflow as tf
import keras

from irm2fl.utils.Metrics import MS_SSIM_POWER_FACTORS

### FUNCTIONS

def mean_absolute_error(y_true, y):
    return tf.reduce_mean(tf.math.abs(y_true - y))

def mean_squared_error(y_true, y):
    return tf.reduce_mean(tf.math.square(y_true - y))

def loss_ssim_multiscale(x, y, max_val=1.0, power_factors=MS_SSIM_POWER_FACTORS[0], filter_size=11, k2=None, **kwargs):
    l_ms_ssim = 1.0 - tf.image.ssim_multiscale(x, y, max_val=max_val, power_factors=power_factors, filter_size=filter_size, k2=k2)
    return tf.where(tf.math.is_nan(l_ms_ssim), 2.0, l_ms_ssim)


### CLASSES

class MS_SSIM(keras.losses.LossFunctionWrapper):
    def __init__(self, name=None, max_val=1.0, n_scales=1, filter_size=11, k2=0.06):
        super().__init__(loss_ssim_multiscale, name=f'{n_scales}S-SSIM' if name is None else name, reduction=keras.utils.losses_utils.ReductionV2.AUTO,
                         max_val=max_val, power_factors=MS_SSIM_POWER_FACTORS[n_scales-1], filter_size=filter_size, k2=k2)

class MSE(keras.losses.LossFunctionWrapper):
    def __init__(self):
        super().__init__(mean_squared_error, name='MSE', reduction=keras.utils.losses_utils.ReductionV2.AUTO)

class MAE(keras.losses.LossFunctionWrapper):
    def __init__(self):
        super().__init__(mean_absolute_error, name='MAE', reduction=keras.utils.losses_utils.ReductionV2.AUTO)
