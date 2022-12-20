import tensorflow as tf
from keras import backend

class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class BaseDiscriminator:

    def __init__(self,
                 name='Disc',
                 final_activation=None,
                 input_shape=None,
                 normalisation='instance',
                 clip_constraint=False):

        self.name = name
        self.final_activation = final_activation
        self.input_shape = input_shape
        self.normalisation = normalisation
        self.clip_constraint = clip_constraint


    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        return self.net(inputs)

    def net(self, inputs):
        raise NotImplementedError