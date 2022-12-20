import irm2fl.data.Augmentation as Aug

def batches2dict(batch1, batch2=None):
    if batch2 is None:
        return batch1
    else:
        return {**batch1, **batch2}


AUGMENTATIONS_VAL_DEFAULT = [
    Aug.get_initial_sample(feature_keys=['image_irm', 'image_if_paired', 'image_if_unpaired'],
                           image_shape={'image_irm': (196, 196, 3),
                                        'image_if_paired': (196, 196, 1),
                                        'image_if_unpaired': (196, 196, 1)}),
    Aug.average(feature_keys=['image_irm']),
    #Aug.resize(output_shape=(128,128)),
    Aug.zoom_and_crop(crop_size=(128, 128), mode=0, probability=0)
]

AUGMENTATIONS_VAL_DEFAULT_3D = [i for n,i in enumerate(AUGMENTATIONS_VAL_DEFAULT) if n!=1]



AUGMENTATIONS_TRAIN_DEFAULT = [
    Aug.get_initial_sample(feature_keys=['image_irm', 'image_if_paired', 'image_if_unpaired'],
                           image_shape={'image_irm': (196, 196, 3),
                                        'image_if_paired': (196, 196, 1),
                                        'image_if_unpaired': (196, 196, 1)}),
    Aug.random_smoothing(feature_keys=['image_if_paired', 'image_if_unpaired'], sigma=2, probability=1),
    Aug.average(feature_keys=['image_irm']),
    #Aug.resize(output_shape=(128,128)),
    Aug.zoom_and_crop(crop_size=(128, 128), mode=1, probability=0.3),
    Aug.fliplr(probability=0.2),
    Aug.flipud(probability=0.2),
    Aug.rot90(probability=0.2),
    Aug.random_contrast(feature_keys=['image_irm'], limits=(1.2, 2), probability=0.05),
    Aug.random_brightness(feature_keys=['image_irm'], factor=0.4, probability=0.05),
    Aug.elastic(feature_keys=['image_irm', 'image_if_paired', 'image_if_unpaired'], amount=10, probability=0.1),
]

AUGMENTATIONS_TRAIN_DEFAULT_3D = [i for n,i in enumerate(AUGMENTATIONS_TRAIN_DEFAULT) if n!=1]
