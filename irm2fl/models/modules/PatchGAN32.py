# code adapted from:  https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/pix2pix/src/model/models.py


import numpy as np

from keras.models import Model
from keras.layers import Conv2D, Input, Concatenate, LeakyReLU, BatchNormalization, \
                         Dense,  Lambda, Reshape, Flatten

from tensorflow_addons.layers import InstanceNormalization

import keras.backend as K

from irm2fl.models.modules import BaseDiscriminator
from irm2fl.models.modules.BaseDiscriminator import ClipConstraint

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)
    return x

def lambda_output(input_shape):
    return input_shape[:2]

def get_list_patches(x, input_shape=(128, 128, 1), patch_size=(32, 32, 1), data_format="channels_last"):
    if data_format == "channels_first":
        h, w = input_shape[1:]
        ph, pw = patch_size[1:]
    else:
        h, w = input_shape[:-1]
        ph, pw = patch_size[:-1]

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if data_format == "channels_last":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(x)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(x)
            list_gen_patch.append(x_patch)

    return list_gen_patch


class PatchGAN32(BaseDiscriminator):

    def __init__(self,
                 name='PatchGAN32',
                 final_activation=None,
                 input_shape=None,
                 normalisation='instance',
                 clip_constraint=False,
                 use_mbd=False):

        super(PatchGAN32, self).__init__(name=name, final_activation=final_activation, clip_constraint=clip_constraint,
                                         normalisation=normalisation, input_shape=input_shape)

        self.use_mbd = use_mbd ### whether to use mini batch discrimination

    def net(self, inputs, data_format="channels_last"):

        patch_size = (32,32,inputs.shape[-1]) if data_format=='channels_last' else (inputs.shape[-3],32,32)

        list_input = get_list_patches(inputs, input_shape=self.input_shape, patch_size=patch_size, data_format=data_format)

        bn_axis = 1 if data_format == "channels_first" else -1

        if self.normalisation=='batch':
            Normalization = BatchNormalization
        elif self.normalisation=='instance':
            Normalization = InstanceNormalization
        else:
            Normalization = None

        kernel_constraint = ClipConstraint(0.01) if self.clip_constraint else None

        def PatchGAN():
            nb_filters = 64
            nb_conv = int(np.floor(np.log(patch_size[1]) / np.log(2)))
            list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

            # First conv
            x_input = Input(shape=patch_size, name="discriminator_input")
            x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
                       padding="same", kernel_constraint=kernel_constraint)(x_input)
            if not Normalization is None: x = Normalization(axis=bn_axis)(x)
            x = LeakyReLU(0.2)(x)

            # Next convs
            for i, f in enumerate(list_filters[1:]):
                name = "disc_conv2d_%s" % (i + 2)
                x = Conv2D(f, (3, 3), strides=(2, 2), name=name,
                           padding="same", kernel_constraint=kernel_constraint)(x)
                if not Normalization is None: x = Normalization(axis=bn_axis)(x)
                x = LeakyReLU(0.2)(x)

            x_flat = Flatten()(x)
            x = Dense(1, activation=self.final_activation, name="disc_dense")(x_flat)

            return Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN")

        module = PatchGAN()

        x = [module(patch)[0] for patch in list_input]
        x_mbd = [module(patch)[1] for patch in list_input]

        if len(x) > 1:
            x = Concatenate(axis=bn_axis)(x)
        else:
            x = x[0]

        if self.use_mbd:
            if len(x_mbd) > 1:
                x_mbd = Concatenate(axis=bn_axis)(x_mbd)
            else:
                x_mbd = x_mbd[0]

            num_kernels = 100
            dim_per_kernel = 5

            M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
            MBD = Lambda(minb_disc, output_shape=lambda_output)

            x_mbd = M(x_mbd)
            x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
            x_mbd = MBD(x_mbd)
            x = Concatenate(axis=bn_axis)([x, x_mbd])

        x_out = Dense(1, activation=self.final_activation, name="disc_output")(x)

        return x_out

