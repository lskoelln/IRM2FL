import tensorflow as tf
from keras.layers import Activation

class BaseGenerator:

    def __init__(self,
                 name="UNet",
                 final_averaging=False,
                 final_activation=None,
                 ndim=2,
                 input_shape=(128,128,1)):

        ### 'silent'
        self.data_format = 'channels_last'

        self.name = name
        self.final_averaging = final_averaging
        self.final_activation = final_activation
        self.ndim = ndim
        self.input_shape = input_shape


    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):

        outputs = self.net(inputs)

        if self.final_averaging:
            axis = 0 if self.data_format == 'channels_first' else -1
            if self.ndim==3:
                _axis = 2 if axis==0 else -2
            elif self.ndim==2:
                _axis = 1 if axis==0 else -1
            outputs = tf.reduce_mean(outputs, axis=_axis)
            outputs = tf.expand_dims(outputs, axis=_axis)

        if not self.final_activation is None:
            outputs = Activation(self.final_activation)(outputs)

        return outputs

    def net(self, **kwargs):
        raise NotImplementedError