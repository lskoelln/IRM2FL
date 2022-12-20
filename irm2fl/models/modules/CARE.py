# code adapted from: https://github.com/CSBDeep/CSBDeep

from irm2fl.models.modules import BaseGenerator
from keras.layers import Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, \
                         concatenate, Add, Activation


def conv_block(n_filter, kernel_size, dropout=0.0, activation="relu", **kwargs):
    conv = Conv2D if len(kernel_size )==2 else Conv3D
    def _func(lay):
        s = conv(n_filter, kernel_size, padding="same", kernel_initializer="glorot_uniform",
                 activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s
    return _func

def unet_block(n_dim, n_filter, kern_size):
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D
    channel_axis = -1 ### could also be 1
    def _func(inputs):
        skip_layers = []
        layer = inputs
        kernel = (kern_size,) * n_dim
        pool = (2,) * n_dim
        # down ...
        for n in range(2):
            for i in range(2):
                layer = conv_block(n_filter * 2 ** n, kernel, name="down_level_%s_no_%s" % (n, i))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name="max_%s" % n)(layer)
        # middle
        layer = conv_block(n_filter * 2 ** 2, kernel, name="middle_%s" % 0)(layer)
        layer = conv_block(n_filter * 2, kernel, name="middle_%s" % 1)(layer)
        # ...and up with skip layers
        for n in reversed(range(2)):
            layer = concatenate([upsampling(pool)(layer), skip_layers[n]], axis=channel_axis)
            layer = conv_block(n_filter * 2 ** n, kernel, name="up_level_%s_no_%s" % (n, 0))(layer)
            layer = conv_block(n_filter * 2 ** max(0,n-1), kernel,
                               activation='linear',
                               name="up_level_%s_no_%s" % (n, 1))(layer)
        return layer
    return _func


class CARE(BaseGenerator):

    def __init__(self, name='CARE', final_averaging=False, final_activation=None, ndim=2,
                 input_shape=(128, 128, 1), n_filter=32):
        super(CARE, self).__init__(name=name, final_averaging=final_averaging, final_activation=final_activation,
                                   ndim=ndim, input_shape=input_shape)
        self.n_filter = n_filter

    def net(self, inputs):

        kern_size = 3 if self.ndim == 3 else 5

        conv = Conv2D if self.ndim == 2 else Conv3D

        unet  = unet_block(self.ndim, self.n_filter, kern_size)(inputs)

        outputs = conv(self.input_shape[-1], (1, )*self.ndim, activation='linear')(unet)
        outputs = Add()([outputs, inputs])
        outputs = Activation(activation="linear")(outputs)

        return outputs
