import tensorflow as tf
import os

from keras.models import Model

from irm2fl.utils import generate_folder
from irm2fl.models.GANLoss import GANLoss

class BaseModel(tf.keras.Model):

    def __init__(self,
                 name='BaseModel',
                 gan_mode=None,
                 alpha=None,
                 generator=None,
                 generator2=None,
                 discriminator=None,
                 discriminator2=None,
                 input_name='image_irm',
                 target_name_paired='image_if_paired',
                 target_name_unpaired='image_if_unpaired',
                 dir_model='my_model'):

        name = '_'.join([str(i) for i in [name, alpha, gan_mode] if not i is None])

        super(BaseModel, self).__init__(name=name)

        self.built = True

        self.dir_model = dir_model

        self.alpha = alpha
        self.gan_mode = gan_mode
        self.gan_loss = GANLoss(gan_mode=gan_mode) if not gan_mode is None else None

        self.input_name = input_name
        self.target_name_paired = target_name_paired
        self.target_name_unpaired = target_name_unpaired

        self.generator = generator
        self.generator2 = generator2
        self.discriminator = discriminator
        self.discriminator2 = discriminator2

        self.model_gen   = self.create_individual_model(self.generator, name='gen')
        self.model_gen2  = self.create_individual_model(self.generator2, name='gen2')
        self.model_disc  = self.create_individual_model(self.discriminator, name='disc')
        self.model_disc2 = self.create_individual_model(self.discriminator2, name='disc2')

    def __call__(self, inputs, training=False):
        return self.call(inputs, training=training)

    def create_individual_model(self, net_block, name='model'):
        if net_block is None:
            return None
        else:
            input_shape = net_block.input_shape
            inputs = tf.keras.layers.Input(input_shape)
            outputs = net_block(inputs)
            return Model(inputs, outputs, name=name)

    def test_step(self, data):
        if type(data) is dict:
            return self.train_step(data, training=False)

        inputs_irm, targets_if_paired = data
        outputs_gen = self(inputs_irm)
        self.compiled_metrics.update_state(targets_if_paired, outputs_gen)

        return {m.name: m.result() for m in self.metrics}

    def save_weights(self, file='weights.h5', overwrite=True, options=None):
        generate_folder(os.path.join(self.dir_model, 'weights'))
        print('Weights are saved.')
        for model in [i for i in [self.model_gen, self.model_gen2, self.model_disc, self.model_disc2] if not i is None]:
            model.save_weights(os.path.join(self.dir_model, r'weights/{}_{}'.format(model.name, file)),
                               overwrite=overwrite, options=options)

    def load_weights(self, file=None, **kwargs):
        if file is None:
            last_epoch = [int(f.split("_")[-1].replace(".h5", "")) for f in os.listdir(os.path.join(self.dir_model, 'weights')) \
                          if all([i in f for i in ['.h5', 'weights']])]
            last_epoch = max(last_epoch)
            basefile = f"weights_{last_epoch}.h5"
        else:
            last_epoch = None
            basefile = file
        print(f"Weights loaded into model ({basefile}).")
        for model in [i for i in [self.model_gen, self.model_gen2, self.model_disc, self.model_disc2] if not i is None]:
            model.load_weights(os.path.join(self.dir_model, r'weights/{}_{}'.format(model.name, basefile)), **kwargs)
        return last_epoch

    def save(self, *args, **kwargs): ### save and save_model are the same
        #kwargs['include_optimizer']=True
        for model in [i for i in [self.model_gen, self.model_gen2, self.model_disc, self.model_disc2] if not i is None]:
            model.save(os.path.join(self.dir_model, r'saved/{}'.format(model.name)), *args, **kwargs)
