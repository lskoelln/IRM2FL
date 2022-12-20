## code adapted from: https://www.tensorflow.org/tutorials/generative/pix2pix

import tensorflow as tf

from irm2fl.models.BaseModel import BaseModel
import irm2fl.models.Losses as Losses
import irm2fl.models.utils_models as utils_models

class Pix2Pix(BaseModel):

    def __init__(self,
                 name='Pix2Pix',
                 alpha=10,
                 gan_mode='ls',
                 generator=None,
                 discriminator=None,
                 input_name='image_irm',
                 target_name_paired='image_if_paired',
                 dir_model='my_model'):

        super(Pix2Pix, self).__init__(name=name, gan_mode=gan_mode, alpha=alpha,
                                      generator=generator, generator2=None,
                                      discriminator=discriminator, discriminator2=None,
                                      input_name=input_name, target_name_paired=target_name_paired,
                                      target_name_unpaired=None, dir_model=dir_model)

        self.loss_tracker_g = tf.keras.metrics.Mean(name="loss")
        self.metric_tracker_g_1 = tf.keras.metrics.Mean(name="gen_loss")
        self.metric_tracker_g_2 = tf.keras.metrics.Mean(name="gen_gan_loss")
        self.metric_tracker_d = tf.keras.metrics.Mean(name="disc_loss")


    def call(self, inputs, training=False):

        if training or type(inputs) is list: ### for training+test

            inputs_irm, targets_if_paired = inputs

            outputs_gen = self.model_gen(inputs_irm, training=training)

            inputs_irm = tf.reduce_mean(inputs_irm, axis=-1) ### in case, this is a multi-channel image
            inputs_irm_disc_real = tf.expand_dims(tf.identity(inputs_irm), axis=-1)
            inputs_irm_disc_fake = tf.expand_dims(tf.identity(inputs_irm), axis=-1)

            inputs_disc_real = tf.concat([inputs_irm_disc_real, targets_if_paired], axis=-1)
            inputs_disc_fake = tf.concat([inputs_irm_disc_fake, tf.identity(outputs_gen)], axis=-1)

            outputs_disc_real = self.model_disc(inputs_disc_real, training=training)
            outputs_disc_fake = self.model_disc(inputs_disc_fake, training=training)

            return outputs_gen, outputs_disc_real, outputs_disc_fake

        else:  ### for evaluate+predict where input is IRM image
            return self.model_gen(inputs, training=training)


    def compile(self, lr=5e-4, loss=None, run_eagerly=None, metrics=None, optimiser='Adam', **kwargs):

        self.g_loss = loss if not loss is None else Losses.MAE()
        self.g_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_gen.compile(loss=self.g_loss, optimizer=self.g_optimizer, run_eagerly=run_eagerly)

        self.d_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_disc.compile(optimizer=self.d_optimizer, run_eagerly=run_eagerly)

        super(Pix2Pix, self).compile(metrics=metrics, run_eagerly=run_eagerly)


    def train_step(self, data, training=True):

        inputs_irm = data[self.input_name]
        targets_if_paired = data[self.target_name_paired]

        if training:

            with tf.GradientTape(persistent=True) as tape:
                outputs_gen, outputs_disc_real, outputs_disc_fake = self.call([inputs_irm, targets_if_paired], training=True)

                gen_loss = self.g_loss(targets_if_paired, outputs_gen)
                gen_gan_loss = self.gan_loss(outputs_disc_fake, target_is_real=True)
                gen_total_loss = gen_gan_loss + self.alpha * gen_loss

                disc_loss = self.gan_loss(outputs_disc_real, target_is_real=True) \
                            + self.gan_loss(outputs_disc_fake, target_is_real=False)

            gen_gradients = tape.gradient(gen_total_loss, self.model_gen.trainable_variables)
            disc_gradients = tape.gradient(disc_loss, self.model_disc.trainable_variables)

            self.g_optimizer.apply_gradients(zip(gen_gradients, self.model_gen.trainable_variables))
            self.d_optimizer.apply_gradients(zip(disc_gradients, self.model_disc.trainable_variables))

        else:
            outputs_gen, outputs_disc_real, outputs_disc_fake = self.call([inputs_irm, targets_if_paired], training=False)
            gen_loss = self.g_loss(targets_if_paired, outputs_gen)
            gen_gan_loss = self.gan_loss(outputs_disc_fake, target_is_real=True)
            gen_total_loss = gen_gan_loss + self.alpha * gen_loss

            disc_loss = self.gan_loss(outputs_disc_real, target_is_real=True) \
                        + self.gan_loss(outputs_disc_fake, target_is_real=False)

        self.loss_tracker_g.update_state(gen_total_loss)
        self.metric_tracker_g_1.update_state(gen_loss)
        self.metric_tracker_g_2.update_state(gen_gan_loss)
        self.metric_tracker_d.update_state(disc_loss)

        self.compiled_metrics.update_state(targets_if_paired, outputs_gen)

        return {m.name: m.result() for m in self.metrics}