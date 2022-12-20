# code adapted from: https://www.tensorflow.org/tutorials/generative/pix2pix

from irm2fl.models.modules import BaseDiscriminator
from irm2fl.models.modules.BaseDiscriminator import ClipConstraint

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization

import numpy as np


def downsample(filters, size, norm='batch', kernel_constraint=None):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False, kernel_constraint=kernel_constraint))
    if norm=='batch':
        result.add(BatchNormalization())
    elif norm=='instance':
        result.add(InstanceNormalization())
    result.add(LeakyReLU())
    return result

def W_init(shape, name=None, **kwargs):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def b_init(shape, name=None, **kwargs):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


class PatchGAN34(BaseDiscriminator):

    def __init__(self,
                 name='PatchGAN34',
                 final_activation=None,
                 normalisation='instance',
                 clip_constraint=False,
                 input_shape=None):

        super(PatchGAN34, self).__init__(name=name, final_activation=final_activation, clip_constraint=clip_constraint,
                                      normalisation=normalisation, input_shape=input_shape)

    def net(self, inputs):

        if self.normalisation=='batch':
            Normalization = BatchNormalization
        elif self.normalisation=='instance':
            Normalization = InstanceNormalization
        else:
            Normalization = None

        kernel_constraint = ClipConstraint(0.01) if self.clip_constraint else None

        initializer = tf.random_normal_initializer(0., 0.02)

        x = inputs
        x = downsample(64, 4, norm=None, kernel_constraint=kernel_constraint)(x)
        x = downsample(128, 4, norm=self.normalisation, kernel_constraint=kernel_constraint)(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = tf.keras.layers.Conv2D(256, 4, strides=1, kernel_initializer=initializer, use_bias=False,
                                   kernel_constraint=kernel_constraint)(x)
        if not Normalization is None: x = Normalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.ZeroPadding2D()(x)
        x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, activation=self.final_activation,
                                   kernel_constraint=kernel_constraint)(x)
        return x
