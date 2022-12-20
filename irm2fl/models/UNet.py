import tensorflow as tf

from irm2fl.models.BaseModel import BaseModel
import irm2fl.models.utils_models as utils_models

class UNet(BaseModel):

    def __init__(self,
                 name='UNet',
                 generator=None,
                 dir_model='my_model',
                 input_name='image_irm',
                 target_name_paired='image_if_paired'):

        super(UNet, self).__init__(name=name, gan_mode=None, alpha=None,
                                   generator=generator, generator2=None,
                                   discriminator=None, discriminator2=None,
                                   input_name=input_name, target_name_paired=target_name_paired,
                                   target_name_unpaired=None, dir_model=dir_model)

    def call(self, inputs, training=False):
        return self.model_gen(inputs, training=training)

    def compile(self, lr=5e-4, loss=None, run_eagerly=None, metrics=None, optimiser='Adam', **kwargs):
        optimizer = getattr(utils_models, optimiser)(lr=lr)
        super(UNet, self).compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=run_eagerly)

    def train_step(self, data, training=True):

        #### code adapted from: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

        inputs_irm = data[self.input_name]
        targets_if_paired = data[self.target_name_paired]

        if training:
            with tf.GradientTape() as tape:
                outputs_gen = self.call(inputs_irm, training=True)
                loss = self.compiled_loss(targets_if_paired, outputs_gen, regularization_losses=self.losses)

            # Compute gradients
            gradients = tape.gradient(loss, self.model_gen.trainable_variables)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, self.model_gen.trainable_variables))

        else:
            outputs_gen = self.call(inputs_irm, training=True)
            self.compiled_loss(targets_if_paired, outputs_gen, regularization_losses=self.losses)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(targets_if_paired, outputs_gen)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
