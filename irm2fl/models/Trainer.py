import tensorflow as tf
import copy
import os

import matplotlib.pyplot as plt

from irm2fl.data.TFRecords import get_dataset
from irm2fl.data import AUGMENTATIONS_VAL_DEFAULT, AUGMENTATIONS_TRAIN_DEFAULT, batches2dict
import irm2fl.utils.Metrics as Metrics

from irm2fl.utils.utils_history import export_history
from irm2fl.utils import generate_folder

#### TODO:
## check if deepcopy of augmentations necessary
## get rid of loss in evaluation
## load best weights

class Trainer:

    def __init__(self,
                 data=None,
                 model = None,
                 aug_train=None,
                 aug_val=None,
                 val_split=0.1,
                 batch_size=16
                 ):

        self.datas = data if type(data) is list else [data]
        self.model = model
        
        if aug_train is None:
            self.aug_train = [AUGMENTATIONS_TRAIN_DEFAULT[0]]+AUGMENTATIONS_TRAIN_DEFAULT[2:] if 'UNet' in self.model.name \
                                                                                               else AUGMENTATIONS_TRAIN_DEFAULT
        else:
            self.aug_train = AUGMENTATIONS_VAL_DEFAULT if aug_train is False else aug_train
            
        self.aug_val = AUGMENTATIONS_VAL_DEFAULT if aug_val is None else aug_val

        self.batch_size = batch_size
        self.val_split = val_split

    def get_zipped_dataset_train(self, batch_size=None, augmentations=None, shuffle_size=None):
        if batch_size is None: batch_size = self.batch_size
        if augmentations is None: augmentations=self.aug_train
        return tf.data.Dataset.zip(tuple([get_dataset(i.get_filenames_train(self.val_split), batch_size,
                                    augmentations=[i.parse_tfrecord_fn] + copy.deepcopy(augmentations),
                                    shuffle_size=shuffle_size) for i in self.datas])).map(batches2dict)

    def get_zipped_dataset_val(self, batch_size=None, augmentations=None, shuffle_size=None):
        if batch_size is None: batch_size = self.batch_size
        if augmentations is None: augmentations=self.aug_val
        return tf.data.Dataset.zip(tuple([get_dataset(i.get_filenames_val(self.val_split), batch_size,
                                    augmentations=[i.parse_tfrecord_fn] + copy.deepcopy(augmentations),
                                    shuffle_size=shuffle_size) for i in self.datas])).map(batches2dict)

    def get_steps_per_epoch_train(self, batch_size=None):
        if batch_size is None: batch_size=self.batch_size
        return [n for n, i in enumerate(tf.data.Dataset.zip(tuple([get_dataset(i.get_filenames_train(self.val_split), batch_size,
                augmentations=[i.parse_tfrecord_fn], shuffle_size=1) for i in self.datas])))][-1]

    def get_steps_per_epoch_val(self, batch_size=None):
        if batch_size is None: batch_size=self.batch_size
        return [n for n, i in enumerate(tf.data.Dataset.zip(tuple([get_dataset(i.get_filenames_val(self.val_split), batch_size,
                augmentations=[i.parse_tfrecord_fn], shuffle_size=1) for i in self.datas])))][-1]

    def train(self,
              epochs=1,
              steps_per_epoch=None,
              lr=1e-5,
              loss=None,
              optimiser='RMSprop',
              metrics=None,
              save_freq=5,
              load_weights=True):

        self.plot_traig_data(file_out='example_before-traig.png')

        ### load_weights can be booleans or weight file

        ### DATASETS
        steps_per_epoch_train = self.get_steps_per_epoch_train() if steps_per_epoch is None else steps_per_epoch
        steps_per_epoch_val = self.get_steps_per_epoch_val() if steps_per_epoch is None else int(self.val_split*steps_per_epoch)

        initial_epoch = 0
        if not load_weights is False:
            try:
                initial_epoch = self.model.load_weights(load_weights) if type(load_weights) is str else self.model.load_weights()
                initial_epoch += 1
                epochs -= initial_epoch
            except:
                print("Training is conducted without prior loading of weights.")

        if epochs==0:
            return

        ### MODEL
        self.model.compile(lr=lr, loss=loss, metrics=metrics, optimiser=optimiser)
        save_freq = epochs if save_freq is None or save_freq>=epochs else save_freq

        for n in range(int(epochs/save_freq)):
            self.model.fit( x = self.get_zipped_dataset_train().repeat(epochs),
                            epochs = (n+1)*save_freq+initial_epoch,
                            steps_per_epoch = steps_per_epoch_train,
                            verbose = 1,
                            validation_data = self.get_zipped_dataset_val().repeat(epochs),
                            validation_steps = steps_per_epoch_val,
                            initial_epoch = n*save_freq+initial_epoch )
            last_epoch = self.model.history.epoch[-1]
            self.model.save_weights(f'weights_{last_epoch}.h5')
            export_history(self.model.dir_model, self.model.history)
            self.plot_examples(file_ext=last_epoch)

    def plot_examples(self, no_examples=5, file_ext=None, file_out=None, shuffle_size=1, display=False):

        generate_folder(self.model.dir_model)

        file_out = os.path.join(self.model.dir_model, f"examples{'' if file_ext is None else '_%s' % file_ext}.png" \
                                                                    if file_out is None else file_out)

        for i, example in enumerate(self.get_zipped_dataset_val(batch_size=1, shuffle_size=shuffle_size)):

            x_input = example[self.model.input_name]
            x_targets = { n.split('image_')[-1]: example[n] for n in [self.model.target_name_paired, self.model.target_name_unpaired] \
                                                         if not n is None }
            x_preds = { f'pred ({m.name})': m(x_input if n==0 else self.model.model_gen(x_input)) \
                                            for n,m in enumerate([self.model.model_gen, self.model.model_gen2]) if not m is None }

            plotdata = {self.model.input_name.split('image_')[-1]: x_input, **x_preds, **x_targets}

            if i==0:
                plt.figure(figsize=(10,1.9*len(plotdata)))
                cols = no_examples
                rows = 1+len(x_preds)+len(x_targets)

            for k, key in enumerate(plotdata):
                x = plotdata[key]
                plt.subplot(rows,cols,k*5+1+i); plt.imshow(x[0], cmap="magma" if x.shape[-1]==3 else "gray");
                plt.xticks([]); plt.yticks([])
                if i==0: plt.ylabel(key)

            if i==no_examples-1:
                break

        plt.tight_layout()
        plt.show() if display else plt.savefig(file_out); plt.close()


    def plot_traig_data(self, no_examples=5, file_ext=None, file_out=None, shuffle_size=1, display=False):

        generate_folder(self.model.dir_model)

        file_out = os.path.join(self.model.dir_model, f"traigs_data{'' if file_ext is None else '_%s' % file_ext}.png" \
                                                                     if file_out is None else file_out)

        for i, example in enumerate(self.get_zipped_dataset_train(batch_size=1, shuffle_size=shuffle_size)):

            x_input = example[self.model.input_name]
            x_targets = { n: example[n] for n in [self.model.target_name_paired, self.model.target_name_unpaired] if not n is None }

            if i==0:
                plt.figure(figsize=(10,1.9*2))
                cols = no_examples
                rows = 1+len(x_targets)
            plt.subplot(rows,cols,1+i); plt.imshow(x_input[0], cmap="magma" if x_input.shape[-1]==3 else "gray");
            plt.xticks([]); plt.yticks([])
            if i==0: plt.ylabel(self.model.input_name.split('_')[-1])
            for t, key in enumerate(x_targets):
                plt.subplot(rows,cols,6+cols*t+i); plt.imshow(x_targets[key][0], cmap="magma" \
                            if x_targets[key].shape[-1]==3 else "gray"); plt.xticks([]); plt.yticks([])
                if i==0: plt.ylabel(key.split('image_')[-1])
            if i==no_examples-1:
                break
        plt.tight_layout()
        if display:
            plt.show()
        else:
            plt.savefig(file_out)
            plt.close()


    def evaluate(self, file=None, mean=True, test_dataset=None, load_weights=True, batch_size=16, overwrite=True):

        if load_weights:
            last_epoch = self.model.load_weights(file=load_weights) if type(load_weights) is str else self.model.load_weights()
        else:
            last_epoch = None

            
        if file is None:
            file = f"evaluation{'_mean' if mean is True else ''}{'_%s' % last_epoch if not last_epoch is None else ''}.txt"
            
        if os.path.isfile(os.path.join(self.model.dir_model, file)) and overwrite is False:
            return
            
        generate_folder(self.model.dir_model)

        if batch_size is None:
            batch_size = 1

        self.model.compile( run_eagerly=True, metrics=[
                                Metrics.PSNR(),
                                Metrics.NRMSE(),
                                Metrics.NCC(),
                                Metrics.MS_SSIM(n_scales=1, name='SSIM', crop=True),
                                Metrics.MS_SSIM(n_scales=3, name='3S-SSIM', crop=True),
                                Metrics.MS_SSIM(n_scales=5, filter_size=7, name='5S-SSIM', crop=True)
                                                        ] )

        if mean is True:
            dict_evaluation = self.model.evaluate( self.get_zipped_dataset_val(batch_size=batch_size, shuffle_size=1) \
                                                                          if test_dataset is None else test_dataset,
                                                   batch_size=batch_size,
                                                   verbose=2,
                                                   return_dict=True )

        else:
            for idx, batch in enumerate(self.get_zipped_dataset_val(batch_size=batch_size, shuffle_size=1) \
                                                                            if test_dataset is None else test_dataset):
                
                dict_eval = self.model.evaluate( batch[self.model.input_name],
                                                 batch[self.model.target_name_paired],
                                                 batch_size=batch_size,
                                                 verbose=3,
                                                 return_dict=True )
                if idx==0:
                    dict_evaluation = {key: list() for key in ['index']+list(dict_eval.keys())}
                dict_evaluation['index'].append(idx)
                for key in dict_eval.keys():
                    dict_evaluation[key].append(dict_eval[key])
                if type(mean) is int:
                    if idx==mean:
                        break

        keys = list(dict_evaluation.keys())

        with open(os.path.join(self.model.dir_model, file), 'w') as file:
            file.write("\t".join(keys)+'\n')
            if type(dict_evaluation[keys[0]]) is list:
                for n in range(len(dict_evaluation[keys[0]])):
                    export = [str(dict_evaluation[key][n]) for key in keys]
                    file.write("\t".join(export)+'\n')
            else:
                export = [str(dict_evaluation[key]) for key in keys]
                file.write("\t".join(export) + '\n')
