import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

def Adam(lr=1e-5):
    return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

def RMSprop(lr=1e-5, **kwargs):
    return tf.keras.optimizers.Adam(learning_rate=lr)

def clip_relu(x):
    return tf.keras.activations.relu(x, max_value=1.)

get_custom_objects().update({'clip_relu': tf.keras.layers.Activation(clip_relu)})