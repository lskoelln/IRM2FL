import tensorflow as tf
import irm2fl.models.Losses as Losses
from keras import backend as K

#### code adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9bcef69d5b39385d18afad3d5a839a02ae0b43e7/models/networks.py#L209

class GANLoss:

    def __init__(self, gan_mode='van', target_real_label=1.0, target_fake_label=0.0):
        self.gan_mode = gan_mode
        if gan_mode=='van':
            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif gan_mode=='ls':
            self.loss = Losses.MSE()
        elif gan_mode=='ws':
            self.loss = None
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label

    def __call__(self, prediction, target_is_real):
        if self.gan_mode=='ws':
            if target_is_real:
                return - K.mean(prediction)
            else:
                return K.mean(prediction)
        else:
            target = self.get_target_tensor(prediction, target_is_real)
            return self.loss(target, prediction)

    def get_target_tensor(self, prediction, target_is_real):
        noise = tf.random.normal(prediction.shape, mean=0.0, stddev=0.1) if not None in prediction.shape else 0.0
        if target_is_real:
            return self.target_real_label + tf.zeros_like(prediction) + noise
        else:
            return self.target_fake_label + tf.zeros_like(prediction) + noise
