#implement WGANGP

#######################################################################

from __future__ import print_function, division
from LossFunctions import wasserstein_loss,gradient_penalty_loss
from keras.layers import Input,Softmax
from keras.models import  Model
import keras
from functools import partial
import numpy as np
from RandomWeightedAverage import RandomWeightedAverage
from critic import build_critic
from generator import build_generator
from keras.utils import plot_model
from readData import readH5file,numberOfClass

#######################################################################
class CLSWGANGP():
    def __init__(self):

        ##########################################
        self.features_shape = (2048,)
        self.latent_dim = 85
        self.n_critic = 5
        optimizer = keras.optimizers.Adam(lr=0.00001, beta_1=0.5, beta_2=0.999, amsgrad=False)
        self.generator = build_generator()
        self.critic = build_critic()
        self.batch_size=1024
        self.losslog = []
        self.nclasses=numberOfClass()
        ##########################################

        # Freeze generator's layers while training critic
        self.generator.trainable = False
        # features input (real sample)
        real_features = Input(shape=self.features_shape)
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate features based of noise (fake sample) and add label to the input
        label = Input(shape=(85,))
        fake_features = self.generator([z_disc, label])
        # Discriminator determines validity of the real and fake images
        fake  = self.critic([fake_features, label])
        valid = self.critic([real_features, label])
        # Construct weighted average between real and fake images
        interpolated_features = RandomWeightedAverage()([real_features, fake_features])
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_features, label])
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,averaged_samples=interpolated_features)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_features, label, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[wasserstein_loss    ,
                                        wasserstein_loss    ,
                                        partial_gp_loss]    ,
                                        optimizer=optimizer ,
                                        loss_weights=[1, 1, 10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------
        from keras.models import load_model
        classificationLayer = load_model('./models/classifierLayer.h5')
        classificationLayer.name = 'modelClassifier'
        # For the generator we freeze the critic's layers + classification Layers
        self.critic.trainable = False
        classificationLayer.trainable=False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # add label to the input
        label = Input(shape=(85,))
        # Generate images based of noise
        features = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([features, label])
        # Discriminator determines class
        classx=classificationLayer(features)

        self.generator_model = Model([z_gen, label],[valid,classx])
        plot_model(self.generator_model,to_file="./doc/model.pdf",show_shapes=True)
        self.generator_model.compile(loss=[wasserstein_loss,'binary_crossentropy'],optimizer=optimizer,loss_weights=[1, 0.01])

    def train(self, epochs, batch_size, sample_interval=50):
        (x_train, y_train, a_train), (x_test, y_test, a_test), (x_val, y_val, a_val) = readH5file()
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                features, labels = x_train[idx], a_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([features, labels, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            features, labels, attr = x_train[idx], y_train[idx],a_train[idx]
            import keras
            labels = keras.utils.to_categorical(labels, 50)
            g_loss = self.generator_model.train_on_batch([noise, attr],[valid,labels])

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, np.mean(d_loss), np.mean(g_loss)))


            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.generator.save('./models/generator.h5', overwrite=True)
                self.critic.save('./models/critic.h5', overwrite=True)
