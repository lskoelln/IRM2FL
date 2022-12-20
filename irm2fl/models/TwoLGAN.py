import tensorflow as tf

from irm2fl.models.BaseModel import BaseModel
import irm2fl.models.Losses as Losses
import irm2fl.models.utils_models as utils_models

class TwoLGAN(BaseModel):

    def __init__(self,
                 name='TwoLGAN',
                 gan_mode='ls',
                 alpha=10,
                 generator=None,
                 generator2=None,
                 discriminator=None,
                 input_name='image_irm',
                 target_name_paired='image_if_paired',
                 target_name_unpaired = 'image_if_unpaired',
                 dir_model='my_model'):

        super(TwoLGAN, self).__init__(name=name, gan_mode=gan_mode, alpha=alpha,
                                      generator=generator, generator2=generator2,
                                      discriminator=discriminator, discriminator2=None,
                                      input_name=input_name, target_name_paired=target_name_paired,
                                      target_name_unpaired=target_name_unpaired, dir_model=dir_model)

        self.loss_tracker_g = tf.keras.metrics.Mean(name="loss")
        self.metric_tracker_g = tf.keras.metrics.Mean(name="gen_gan_loss")
        self.metric_tracker_g2 = tf.keras.metrics.Mean(name="gen2_loss")
        self.metric_tracker_d = tf.keras.metrics.Mean(name="disc_loss")

    def call(self, inputs, training=False):

        if training or type(inputs) is list: ### for training+test

            inputs_irm, targets_if_unpaired = inputs

            outputs_gen = self.model_gen(tf.identity(inputs_irm), training=training)

            outputs_disc_real = self.model_disc(targets_if_unpaired, training=training)
            outputs_disc_fake = self.model_disc(tf.identity(outputs_gen), training=training)

            outputs_gen2 = self.model_gen2(tf.identity(outputs_gen), training=training)

            return outputs_gen, outputs_gen2, outputs_disc_real, outputs_disc_fake

        else:  ### for evaluate+predict where input is IRM image
            return self.model_gen(inputs, training=training)


    def compile(self, lr=5e-4, loss=None, run_eagerly=None, metrics=None, optimiser='Adam', **kwargs):

        self.g_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_gen.compile(optimizer=self.g_optimizer, run_eagerly=run_eagerly)

        self.g2_loss = Losses.MAE() if loss is None else loss
        self.g2_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_gen2.compile(loss=self.g2_loss, optimizer=self.g2_optimizer, run_eagerly=run_eagerly)

        self.d_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_disc.compile(optimizer=self.d_optimizer, run_eagerly=run_eagerly)

        super(TwoLGAN, self).compile(metrics=metrics, run_eagerly=run_eagerly)


    def train_step(self, data, training=True):

        inputs_irm = data[self.input_name]
        targets_if_paired = data[self.target_name_paired]
        targets_if_unpaired = data[self.target_name_unpaired]

        if training:

            with tf.GradientTape(persistent=True) as tape:
                outputs_gen, outputs_gen2, outputs_disc_real, outputs_disc_fake = self.call([inputs_irm, targets_if_unpaired], training=True)

                ### generator_loss from Pix2Pix, but is here calculated with generated image and target of gen2
                gen2_loss = self.g2_loss(targets_if_paired, outputs_gen2)
                
                gen_gan_loss = self.gan_loss(outputs_disc_fake, target_is_real=True)
                gen_total_loss = gen_gan_loss + self.alpha * gen2_loss

                disc_loss = self.gan_loss(outputs_disc_real, target_is_real=True) \
                            + self.gan_loss(outputs_disc_fake, target_is_real=False)

            gen2_gradients = tape.gradient(gen2_loss, self.model_gen2.trainable_variables)
            gen_gradients = tape.gradient(gen_total_loss, self.model_gen.trainable_variables)
            disc_gradients = tape.gradient(disc_loss, self.model_disc.trainable_variables)

            self.g2_optimizer.apply_gradients(zip(gen2_gradients, self.model_gen2.trainable_variables))
            self.g_optimizer.apply_gradients(zip(gen_gradients, self.model_gen.trainable_variables))
            self.d_optimizer.apply_gradients(zip(disc_gradients, self.model_disc.trainable_variables))

        else:
            outputs_gen, outputs_gen2, outputs_disc_real, outputs_disc_fake = self.call([inputs_irm, targets_if_unpaired], training=False)
            gen2_loss = self.g2_loss(targets_if_paired, outputs_gen2)
            gen_gan_loss = self.gan_loss(outputs_disc_fake, target_is_real=True)
            gen_total_loss = gen_gan_loss + self.alpha * gen2_loss

            disc_loss = self.gan_loss(outputs_disc_real, target_is_real=True) \
                        + self.gan_loss(outputs_disc_fake, target_is_real=False)

        self.loss_tracker_g.update_state(gen_total_loss)
        self.metric_tracker_g.update_state(gen_gan_loss)
        self.metric_tracker_g2.update_state(gen2_loss)
        self.metric_tracker_d.update_state(disc_loss)

        self.compiled_metrics.update_state(targets_if_paired, outputs_gen)

        return {m.name: m.result() for m in self.metrics}