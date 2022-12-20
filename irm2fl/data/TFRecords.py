import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tifffile import imread

import irm2fl.data.Augmentation as Aug
import irm2fl.data.Normalisation as Normalisation

from irm2fl.data.Preprocessing import Interpolation
from irm2fl.utils import generate_folder
from irm2fl.utils.csbdeep.data import no_background_patches, RawData, create_patches, get_patch_indices, shuffle_inplace


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = tf.convert_to_tensor(value)
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

AUTOTUNE = tf.data.AUTOTUNE


def get_dataset(filenames, batch_size, augmentations=[], shuffle_size=None):
    def prepare_sample(input):
        for func in augmentations:
            input = func(input)
        return input
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
            .map(prepare_sample, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size*20 if shuffle_size is None else shuffle_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(AUTOTUNE)
    )
    return dataset


class TFRecords:

    def __init__( self,
                  basedir     = None,
                  dict_input  =  {'dir_images': None,
                                  'feature_name': None,
                                  'normalisation': Normalisation.NoNorm(),
                                  'patch_size': (192, 192,1)},
                  dict_target =  None,
                  sample_description=u'na',
                  patch_resizing_factor=None,
                  final_pixel_size_in_nm=None,
                  dir_tfrecords=None,
                  dict_feature_encoding=None,
                  dict_feature_decoding=None
                  ):
        #### a few rules:
        ## 'feature_name' must contain 'image'
        ## dict_target can be None
        ## patch_size is tuple with 3 entries ala YXC
        ## sample_description needs to start with u'...'

        #### TODOs:
        ## check if resizing_factor < 1 works

        self.dict_input = dict_input
        self.dict_target = dict_target
        self.dir_tfrecords = dir_tfrecords

        try:
            self.tfr_filenames = [ os.path.join(self.dir_tfrecords, file) for file in os.listdir(self.dir_tfrecords) \
                                    if 'tfrec' in file ]
        except:
            self.tfr_filenames = None
            
        self.basedir = basedir
        
        if not self.dict_input  is None and 'dir_images' in self.dict_input and not self.dict_input['dir_images'] is None:
            self.file_ids = os.listdir(os.path.join(self.basedir, self.dict_input['dir_images']))
        if not self.dict_target is None and 'dir_images' in self.dict_target:
            files_target = os.listdir(os.path.join(self.basedir, self.dict_target['dir_images']))
            self.file_ids = [file for file in self.file_ids if file in files_target]

        self.sample_description = sample_description
        self.patch_resizing_factor = 1 if patch_resizing_factor is None else patch_resizing_factor
        self.final_pixel_size_in_nm = final_pixel_size_in_nm

        if dict_feature_encoding is None:
            self.dict_feature_encoding = {dict_['feature_name']: image_feature for dict_ in [self.dict_input, self.dict_target] if not dict_ is None}
            self.dict_feature_encoding = {**self.dict_feature_encoding,
                                          "file_id": bytes_feature,
                                          "crop_pos_0": int64_feature,
                                          "crop_pos_1": int64_feature,
                                          "resizing_factor": float_feature,
                                          "pixel_size_in_nm": float_feature,
                                          "sample_description": bytes_feature}
        else:
            self.dict_feature_encoding = dict_feature_encoding

        if dict_feature_decoding is None:
            self.dict_feature_decoding = {fn: tf.io.FixedLenFeature([], tf.string) for fn in self.dict_feature_encoding.keys() if 'image' in fn}
            self.dict_feature_decoding = {**self.dict_feature_decoding,
                                          "file_id": tf.io.FixedLenFeature([], tf.string),
                                          "crop_pos_0": tf.io.FixedLenFeature([], tf.int64),
                                          "crop_pos_1": tf.io.FixedLenFeature([], tf.int64),
                                          "resizing_factor": tf.io.FixedLenFeature([], tf.float32),
                                          "pixel_size_in_nm": tf.io.FixedLenFeature([], tf.float32),
                                          "sample_description": tf.io.FixedLenFeature([], tf.string)}
        else:
            self.dict_feature_decoding = dict_feature_decoding


    def create_example(self, example):
        assert example.keys() == self.dict_feature_encoding.keys(), "Keys in example do not match with keys in 'dict_features_func'!"
        feature = {key: self.dict_feature_encoding[key](example[key]) for key in example.keys()}
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_tfrecord_fn(self, example):
        example = tf.io.parse_single_example(example, self.dict_feature_decoding)
        for key in example.keys():
            if 'image' in key:
                example[key] = tf.io.parse_tensor(example[key], out_type=tf.float32)
        return example

    def find_sample_indices(self, invert_lut=True):
        sample_indices = list()
        _dict = self.dict_input if self.dict_target is None else self.dict_target
        for file in self.file_ids:
            img = imread(f"{self.basedir}/{_dict['dir_images']}/{file}")
            assert len(img.shape)==2, "Wrong shape of 'img' - Sample indices are generated from 2-dim image!"
            if invert_lut:
                img=abs(img-1)
            sample_indices.append(get_patch_indices( [img, img], _dict['patch_size'][:2], self.n_patches_per_image,
                                                     datas_mask=None, patch_filter=no_background_patches() ))
        ### sample_indices is list of length 'number of images/files'. Entry of list contains 2 lists for Y, X coordinates - each has the length of n_patches_per_image
        ### C coordinates are added in the following step and set to '-1' which leads to dimension being is left untouched -
        sample_indices = [[n[0], n[1], np.zeros(n[0].shape, dtype=np.int64)-1] for n in sample_indices]
        return sample_indices

    def create(self, n_patches_per_image = 1000, n_tfrecords = 100):

        self.n_patches_per_image = n_patches_per_image
        self.n_tfrecords = n_tfrecords
        self.n_entries_per_tfrecord = int(len(self.file_ids) * self.n_patches_per_image / self.n_tfrecords)

        generate_folder(self.dir_tfrecords)

        if not os.path.exists(self.dir_tfrecords):
            os.makedirs(self.dir_tfrecords)

        #### GET SAMPLE_INDICES
        sample_indices = self.find_sample_indices()

        #### CREATE IMAGE PATCHES AND WRITE TF RECORDS
        raw_data = RawData.from_files( basepath     = self.basedir,
                                       source_files = [os.path.join(self.dict_input['dir_images'], file) for file in self.file_ids],
                                       target_files = None if self.dict_target is None else \
                                                      [os.path.join(self.dict_target['dir_images'], file) for file in self.file_ids],
                                       axes = 'YXC' )

        for itf in range(self.n_tfrecords):

            s = slice(int(self.n_patches_per_image / self.n_tfrecords) * itf, int(self.n_patches_per_image / self.n_tfrecords) * (itf + 1))

            sample_indices_tfr = [[idc[n][s] for n in range(3)] for idc in sample_indices]

            ### Patch dim: SYXC
            X_patches, Y_patches, axes = create_patches(
                                        raw_data       = raw_data,
                                        shuffle        = False,
                                        patch_size     = self.dict_input['patch_size'] if self.dict_target is None else [self.dict_input['patch_size'], self.dict_target['patch_size']],
                                        normalization  = [self.dict_input['normalisation'],
                                                          None if self.dict_target is None else self.dict_target['normalisation']],  ### optional: list
                                        n_patches_per_image = int(self.n_patches_per_image / self.n_tfrecords),
                                        sample_indices = sample_indices_tfr,
                                        verbose        = False,
                                        keep_channel   = True
            )


            sample_indices_tfr = [(n,)+idc for n, item in enumerate(sample_indices_tfr) for idc in zip(*item)]

            shuffle_inplace(X_patches, Y_patches, sample_indices_tfr)

            #### WRITE TFRECORDS
            _interpolate = Interpolation(new_shape=(int(X_patches.shape[1]*self.patch_resizing_factor),
                                                    int(X_patches.shape[2]*self.patch_resizing_factor)),
                                         channel_first=False)

            with tf.io.TFRecordWriter(self.dir_tfrecords + f"/{itf}.tfrec") as writer:

                for p in range(X_patches.shape[0]):

                    x_p, y_p = X_patches[p], Y_patches[p]

                    x_p = _interpolate(x_p)
                    y_p = _interpolate(y_p) if not y_p is None else y_p

                    file_idx, crop_pos_0, crop_pos_1, _ = sample_indices_tfr[p]

                    sample = { dict_['feature_name']: patch for [patch, dict_] in zip([x_p, y_p], [self.dict_input, self.dict_target]) if not dict_ is None }
                    sample = { **sample,
                               "file_id": self.file_ids[file_idx].split('/')[0].encode(),
                               "crop_pos_0": int(crop_pos_0),
                               "crop_pos_1": int(crop_pos_1),
                               "sample_description": self.sample_description.encode('utf-8'),
                               "resizing_factor": float(self.patch_resizing_factor),
                               "pixel_size_in_nm": float(self.final_pixel_size_in_nm)
                               }

                    example = self.create_example(sample)
                    writer.write(example.SerializeToString())

    def get_filenames_val(self, val_split):
        return self.tfr_filenames[:int(np.ceil(len(self.tfr_filenames)*val_split))]

    def get_filenames_train(self, val_split):
        return self.tfr_filenames[int(np.ceil(len(self.tfr_filenames)*val_split)):]

    def get_steps_per_epoch_val(self, val_split=0.1, batch_size=16):
        ds = get_dataset(self.get_filenames_val(val_split), batch_size, augmentations=[self.parse_tfrecord_fn], shuffle_size=1)
        return [n for n, example in enumerate(ds)][-1]

    def get_steps_per_epoch_train(self, val_split=0.1, batch_size=16):
        ds = get_dataset(self.get_filenames_train(val_split), batch_size, augmentations=[self.parse_tfrecord_fn], shuffle_size=1)
        return [n for n, example in enumerate(ds)][-1]

    def check_example(self, index_tfr=0, index_entry=0, augmentations=None):

        file_tfr = os.listdir(self.dir_tfrecords)[index_tfr]

        if augmentations is None:
            aug = [self.parse_tfrecord_fn,
                   Aug.get_initial_sample(feature_keys=None, image_shape={  'image_irm':         (196, 196, 3),
                                                                            'image_if_paired':   (196, 196, 1),
                                                                            'image_if_unpaired': (196, 196, 1) } )]
        else:
            aug = [self.parse_tfrecord_fn] + augmentations

        parsed_dataset = get_dataset(f"{self.dir_tfrecords}/{file_tfr}" , batch_size=1, augmentations=aug, shuffle_size=1)

        for n, features in enumerate(parsed_dataset):

            if n==index_entry:

                for key in features.keys():
                    if not "image" in key:
                        print(f"{key}: {features[key]}")
                        
                print(features.keys())

                file = self.file_ids.index(features['file_id'])
                file = self.file_ids[file]

                x0 = features['crop_pos_0'][0].numpy()
                x1 = features['crop_pos_1'][0].numpy()

                x = features[self.dict_input['feature_name']][0].numpy()

                resizing_factor = features['resizing_factor'][0].numpy()
                step = int((x.shape[1]/resizing_factor)/2)

                X = imread(f"{self.basedir}/{self.dict_input['dir_images']}/{file}")[...,x0 - step:x0 + step, x1 - step:x1 + step]
                X = np.moveaxis(X, 0, -1) if len(X.shape)==3 else np.expand_dims(X,axis=-1)

                plt.figure()
                plt.subplot(1, 2, 1); plt.xticks([]); plt.yticks([]); plt.ylabel('INPUT'); plt.title('Original');
                plt.imshow(X, cmap='gray')
                plt.subplot(1, 2, 2); plt.xticks([]); plt.yticks([]); plt.title(f"Patch{'' if augmentations is None else ' (augm.)'}");
                plt.imshow(x, cmap='gray')
                

                if not self.dict_target is None:
                    Y = imread(f"{self.basedir}/{self.dict_target['dir_images']}/{file}")[...,x0 - step:x0 + step, x1 - step:x1 + step]
                    Y = np.moveaxis(Y, 0, -1) if len(Y.shape) == 3 else Y
                    y = features[self.dict_target['feature_name']][0].numpy()
                    
                    plt.figure()
                    plt.subplot(1, 2, 1); plt.xticks([]); plt.yticks([]); plt.ylabel('TARGET'); plt.title('Original')
                    plt.imshow(Y, cmap='gray' if Y.shape[-1]==1 or len(Y.shape)==2 else None)
                    plt.subplot(1, 2, 2); plt.xticks([]); plt.yticks([]); plt.title(f"Patch{'' if augmentations is None else ' (augm.)'}")
                    plt.imshow(y, cmap='gray' if y.shape[-1]==1 else None)

                return
