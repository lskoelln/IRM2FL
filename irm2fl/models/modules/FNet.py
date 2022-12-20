# code adapted from: https://github.com/AllenCellModeling/pytorch_fnet

from irm2fl.models.modules import BaseGenerator
from keras.layers import BatchNormalization, concatenate, Activation, Conv2D, Conv2DTranspose

class SubNet2Conv():
    def __init__(self, n_in, n_out):
        self.conv1 = Conv2D(n_out, kernel_size=3, padding="same", strides=1)
        self.bn1 = BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.relu1 = Activation(activation='relu')
        self.conv2 = Conv2D(n_out, kernel_size=3, padding="same", strides=1)
        self.bn2 = BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.relu2 = Activation(activation='relu')
    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class _Net_recurse():
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p

        Parameters
        ----------
        in_channels
            Number of channels for input.
        mult_chan
            Factor to determine number of output channels
        depth
            If 0, this subnet will only be convolutions that double the channel
            count.

        """
        self.depth = depth
        n_out_channels = n_in_channels * mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2 * n_out_channels, n_out_channels)
            self.conv_down = Conv2D(n_out_channels, 2, strides=2)
            self.bn0 = BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.relu0 = Activation(activation='relu')
            self.convt = Conv2DTranspose(n_out_channels, kernel_size=2, strides=2)
            self.bn1 = BatchNormalization(epsilon=1e-05, momentum=0.1)
            self.relu1 = Activation(activation='relu')
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
    def __call__(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = concatenate([x_2conv_more, x_relu1], axis=-1)
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class FNet(BaseGenerator):

    def __init__(self, name='FNet', final_averaging=False, final_activation=None, ndim=2,
                 input_shape=(128, 128, 1), n_channels_in=1):
        assert ndim==2, "Error: 'ndim' can only be set to 2 for 'FNet'"
        super(FNet, self).__init__(name=name, final_averaging=final_averaging, final_activation=final_activation,
                                     ndim=ndim, input_shape=input_shape)
        self.n_channels_in=n_channels_in

    def net(self, inputs):

        mult_chan = 32
        depth = 4
        n_in_channels = self.n_channels_in

        def forward(x):
            x_rec = _Net_recurse(n_in_channels=n_in_channels, mult_chan=mult_chan, depth=depth)(x)
            x_rec = Conv2D(1, kernel_size=3, padding="same", strides=1)(x_rec)
            return x_rec

        outputs = forward(inputs)

        return outputs