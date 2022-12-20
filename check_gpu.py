import tensorflow as tf

list_physical_devices = tf.config.list_physical_devices('GPU')

print(list_physical_devices)
