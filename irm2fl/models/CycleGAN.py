import tensorflow as tf

from irm2fl.models.BaseModel import BaseModel
import irm2fl.models.Losses as Losses
import irm2fl.models.utils_models as utils_models

## cycle consistency loss
class calc_cycle_loss:
    def __init__(self, alpha=10, loss=None):
        self.alpha = alpha
        self.loss = Losses.MAE() if loss is None else loss
    def __call__(self, real_image, cycled_image):
        loss_ = self.loss(real_image, cycled_image)
        return self.alpha * loss_

## identity loss
class identity_loss:
    def __init__(self, alpha=10, loss=Losses.MAE()):
        self.alpha=alpha
        self.loss = Losses.MAE() if loss is None else loss
    def __call__(self, real_image, same_image):
        loss_ = self.loss(real_image, same_image)
        return self.alpha * 0.5 * loss_


class CycleGAN(BaseModel):

    def __init__(self,
                 name='CycleGAN',
                 gan_mode = 'ls',
                 alpha=10,
                 generator=None,
                 generator2=None,
                 discriminator=None,
                 discriminator2=None,
                 input_name='image_irm',
                 target_name_unpaired='image_if_unpaired',
                 target_name_paired='image_if_paired',
                 dir_model='my_model'):

        super(CycleGAN, self).__init__(name=name, gan_mode=gan_mode, alpha=alpha,
                                       generator=generator, generator2=generator2,
                                       discriminator=discriminator, discriminator2=discriminator2,
                                       input_name=input_name, target_name_paired=target_name_paired,
                                       target_name_unpaired=target_name_unpaired, dir_model=dir_model)

        self.loss_tracker = tf.keras.metrics.Mean(name="total_cycle_loss")
        self.metric_tracker_g_1 = tf.keras.metrics.Mean(name="gen_loss")
        self.metric_tracker_g_2 = tf.keras.metrics.Mean(name="total_gen_loss")
        self.metric_tracker_g2_1 = tf.keras.metrics.Mean(name="gen2_loss")
        self.metric_tracker_g2_2 = tf.keras.metrics.Mean(name="total_gen2_loss")
        self.metric_tracker_d = tf.keras.metrics.Mean(name="disc_loss")
        self.metric_tracker_d2 = tf.keras.metrics.Mean(name="disc2_loss")

    def call(self, inputs, training=False):
        return self.model_gen(inputs, training=training)

    def compile(self, lr=5e-4, loss=None, run_eagerly=None, metrics=None, optimiser='Adam', **kwargs):

        self.calc_cycle_loss = calc_cycle_loss(alpha=self.alpha, loss=loss)

        self.g_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_gen.compile(optimizer=self.g_optimizer, run_eagerly=run_eagerly)

        self.g2_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_gen2.compile(optimizer=self.g2_optimizer, run_eagerly=run_eagerly)

        self.d_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_disc.compile(optimizer=self.d_optimizer, run_eagerly=run_eagerly)

        self.d2_optimizer = getattr(utils_models, optimiser)(lr=lr)
        self.model_disc2.compile(optimizer=self.d2_optimizer, run_eagerly=run_eagerly)

        super(CycleGAN, self).compile(metrics=metrics, run_eagerly=run_eagerly)


    def train_step(self, data, training=True):

        real_irm = data[self.input_name]
        real_if_unpaired = data[self.target_name_unpaired]
        real_if_paired = data[self.target_name_paired] if self.target_name_paired in data else None ### for evaluation

        with tf.GradientTape(persistent=True) as tape:
            # Generator 1 translates IRM -> FL
            # Generator 2 translates FL -> IRM

            fake_if_unpaired = self.model_gen(real_irm, training=training)
            cycled_irm = self.model_gen2(fake_if_unpaired, training=training)

            fake_irm = self.model_gen2(real_if_unpaired, training=training)
            cycled_if_unpaired = self.model_gen(fake_irm, training=training)

            disc_real_if_unpaired = self.model_disc(real_if_unpaired, training=training)
            disc_fake_if_unpaired = self.model_disc(fake_if_unpaired, training=training)

            disc_real_irm = self.model_disc2(real_irm, training=training)
            disc_fake_irm = self.model_disc2(fake_irm, training=training)

            # calculate the loss
            gen_loss  = self.gan_loss(disc_fake_if_unpaired, target_is_real=True)
            gen2_loss = self.gan_loss(disc_fake_irm, target_is_real=True)

            total_cycle_loss = self.calc_cycle_loss(real_irm, cycled_irm) + self.calc_cycle_loss(real_if_unpaired, cycled_if_unpaired)

            # Total generator loss = adversarial loss + cycle loss
            total_gen_loss  = gen_loss  + total_cycle_loss
            total_gen2_loss = gen2_loss + total_cycle_loss

            disc_loss  = 0.5 * ( self.gan_loss(disc_real_if_unpaired, target_is_real=True) +
                                 self.gan_loss(disc_fake_if_unpaired, target_is_real=False) )

            disc2_loss = 0.5 * ( self.gan_loss(disc_real_irm, target_is_real=True) +
                                 self.gan_loss(disc_fake_irm, target_is_real=False) )

        if training:
            # Calculate the gradients for generator and discriminator
            gen_gradients   = tape.gradient(total_gen_loss , self.model_gen.trainable_variables)
            gen2_gradients  = tape.gradient(total_gen2_loss, self.model_gen2.trainable_variables)
            disc_gradients  = tape.gradient(disc_loss , self.model_disc.trainable_variables)
            disc2_gradients = tape.gradient(disc2_loss, self.model_disc2.trainable_variables)

            # Apply the gradients to the optimizer
            self.g_optimizer. apply_gradients(zip(gen_gradients,   self.model_gen.trainable_variables))
            self.g2_optimizer.apply_gradients(zip(gen2_gradients,  self.model_gen2.trainable_variables))
            self.d_optimizer. apply_gradients(zip(disc_gradients,  self.model_disc.trainable_variables))
            self.d2_optimizer.apply_gradients(zip(disc2_gradients, self.model_disc2.trainable_variables))

        self.loss_tracker.update_state(total_cycle_loss)
        self.metric_tracker_g_1.update_state(gen_loss)
        self.metric_tracker_g_2.update_state(total_gen_loss)
        self.metric_tracker_g2_1.update_state(gen2_loss)
        self.metric_tracker_g2_2.update_state(total_gen2_loss)
        self.metric_tracker_d.update_state(disc_loss)
        self.metric_tracker_d2.update_state(disc2_loss)

        if not real_if_paired is None:
            self.compiled_metrics.update_state(real_if_paired, fake_if_unpaired)

        return {m.name: m.result() for m in self.metrics}


