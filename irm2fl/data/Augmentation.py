import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from augmend import Augmend, AdditiveNoise, Elastic


class get_initial_sample:
    def __init__(self, image_shape={'image_irm': (196, 196, 3), 'image_if_paired': (196, 196, 1),
                                    'image_if_unpaired': (196, 196, 1)},
                 feature_keys=['image_irm', 'image_if_paired', 'image_if_unpaired'] ):
        self.image_shape=image_shape
        self.feature_keys = feature_keys
    def __call__(self, features):
        sample = dict()
        for key in list(features.keys()):
            if self.feature_keys is None or key in self.feature_keys:
                if 'image' in key:
                    features[key].set_shape(self.image_shape[key])
                    sample[key] = tf.image.resize(features[key], size=self.image_shape[key][:2])
                else:
                    sample[key] = features[key]
        return sample

class resize:
    def __init__(self, output_shape=(128,128)):
        self.output_shape = output_shape

    def __call__(self, features):
        sample = dict()
        for key in features.keys():
            if 'image' in key:
                sample[key] = tf.image.resize(features[key], size=self.output_shape)
            else:
                sample[key] = features[key]
        return sample

class average:
    def __init__(self, feature_keys=None, channel_last=True):
        self.feature_keys = feature_keys
        self.channel_last = channel_last

    def __call__(self, features):

        sample = dict()
        for key in features:
            if key in self.feature_keys:
                sample[key] = tf.reduce_mean(features[key], axis=-1 if self.channel_last else 1)
                sample[key] = tf.expand_dims(sample[key], axis=-1 if self.channel_last else 1)
            else:
                sample[key] = features[key]
        return sample

class random_contrast:
    def __init__(self, feature_keys=None, limits=(1.2, 2), probability=0.05, clip=(0,1)):
        self.feature_keys = feature_keys
        self.limits = limits
        self.probability = probability
        self.clip = clip

    def __call__(self, features):

        def call(features):
            sample = dict()
            for key in features:
                if key in self.feature_keys:
                    x = tf.image.random_contrast(features[key], *self.limits)
                    sample[key] = tf.clip_by_value(x, *self.clip)
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features))

class random_brightness:
    def __init__(self, feature_keys=None, factor=0.1, probability=0.05, clip=(0,1)):
        self.feature_keys = feature_keys
        self.factor = factor
        self.probability = probability
        self.clip = clip

    def __call__(self, features):

        def call(features):
            sample = dict()
            for key in features:
                if key in self.feature_keys:
                    x = tf.image.random_brightness(features[key], self.factor)
                    sample[key] = tf.clip_by_value(x, *self.clip)
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features))

class random_smoothing:
    def __init__(self, feature_keys=None, sigma=3, probability=0.05, clip=(0,1)):
        self.feature_keys = feature_keys
        self.sigma = sigma
        self.probability = probability
        self.clip = clip

    def __call__(self, features):

        def call(features, choice):
            sample = dict()
            for key in features:
                if key in self.feature_keys:
                    x = tfa.image.gaussian_filter2d(features[key], filter_shape=(3,3), sigma=self.sigma)
                    sample[key] = tf.clip_by_value(x, *self.clip)
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features, choice))

class fliplr: ### only work with channel last!
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, features):

        def call(features):
            sample = dict()
            for key in features:
                if 'image' in key:
                    sample[key] = tf.image.flip_left_right(features[key])
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features))

class flipud: ### only work with channel last!
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, features):

        def call(features):
            sample = dict()
            for key in features:
                if 'image' in key:
                    sample[key] = tf.image.flip_up_down(features[key])
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features))

class rot90: ### only work with channel last!
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, features):

        def call(features):
            sample = dict()
            for key in features:
                if 'image' in key:
                    sample[key] = tf.image.rot90(features[key])
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond(choice > self.probability, lambda: features, lambda: call(features))

class invert_lut: ### only work with channel last!
    def __init__(self, feature_keys=None):
        self.feature_keys = feature_keys

    def __call__(self, features):

        sample = dict()
        for key in features:
            if key in self.feature_keys:
                sample[key] = tf.abs(features[key]-1)
            else:
                sample[key] = features[key]

        return sample

class zoom_and_crop:

    def __init__(self, probability=1, crop_size=(128,128), factors=(0.5, 1.0), mode=0):
        self.probability = probability
        self.crop_size = crop_size
        self.factors = factors
        self.mode = mode ### 0: (x0, x1) in 'crop_to_bounding_box' are set to (0, 0); 1: (x0, x1) are randomly selected

    def __call__(self, features):

        # Generate crop settings for 'crop_and_resize'
        scales = list(np.arange(self.factors[0], self.factors[1], 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def crop_and_resize(features):
            sample = dict()
            index = tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
            for key in features.keys():
                if 'image' in key:
                    # Create different crops for an image
                    crops = tf.image.crop_and_resize([features[key]], boxes=boxes, box_indices=np.zeros(len(scales)),
                                                     crop_size=self.crop_size)
                    # Return a random crop
                    sample[key] = crops[index]
                else:
                    sample[key] = features[key]
            return sample

        def crop_to_bounding_box(features):
            sample = dict()
            x0 = None
            for k, key in enumerate(features.keys()):
                if 'image' in key:
                    if x0 is None:
                        if self.mode==0:
                            x0, x1 = 0, 0
                        else:
                            x0 = features[key].shape[-3] - self.crop_size[0]
                            x1 = features[key].shape[-2] - self.crop_size[1]
                            x0 = tf.gather(tf.constant(np.arange(0,x0)), tf.cast(tf.random.uniform([]) * x0, tf.int32)) if x0>0 else 0
                            x1 = tf.gather(tf.constant(np.arange(0,x1)), tf.cast(tf.random.uniform([]) * x1, tf.int32)) if x1>0 else 0
                    x0 = tf.cast(x0, tf.int32)
                    x1 = tf.cast(x1, tf.int32)
                    sample[key] = tf.image.crop_to_bounding_box(features[key], x0, x1, self.crop_size[0], self.crop_size[1])
                else:
                    sample[key] = features[key]
            return sample

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        return tf.cond( choice > self.probability,
                        lambda: crop_to_bounding_box(features),
                        lambda: crop_and_resize(features) )

class elastic:

    def __init__(self, probability=0.1, amount=5, order=0, feature_keys=None):
        self.probability = probability
        self.amount = amount
        self.order = order
        self.feature_keys = feature_keys

    def __call__(self, features):

        if not any([i in self.feature_keys for i in features]):
            return features

        aug = Augmend()
        aug.add(Elastic(axis=(1, 2), amount=self.amount, order=self.order), # use_gpu=True
                probability=self.probability)
        augmented_data = aug.tf_map(tf.stack([features[key] for key in features if key in self.feature_keys], axis=0))[0]

        for n, key in enumerate(features):
            if key in self.feature_keys:
                features[key] = augmented_data[n]

        return features


## TODO
# class add_noise:
#
#     def __init__(self, probability=0.1, sigma=0.1, feature_keys=None):
#         self.probability = probability
#         self.sigma = sigma
#         self.feature_keys = feature_keys
#
#     def __call__(self, features):
#
#         if not any([i in self.feature_keys for i in features]):
#             return features
#
#         aug = Augmend()
#         aug.add(AdditiveNoise(sigma=self.sigma), probability=self.probability)
#         augmented_data = aug.tf_map(tf.stack([features[key] for key in features if key in self.feature_keys], axis=0))[0]
#
#         for n, key in enumerate(features):
#             if key in self.feature_keys:
#                 features[key] = augmented_data[n]
#
#         return features
