#implement Generator

#######################################################################

from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Concatenate, Activation,LeakyReLU
from keras.models import Sequential, Model

#######################################################################

def build_generator():

    ##########################################
    latent_dim    = 85
    attribute_dim = 85
    features_dim  = 2048
    in_dim        = attribute_dim + latent_dim
    out_shape     = (features_dim,)
    ##########################################

    model = Sequential()
    model.add(Dense(4096, input_dim=in_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2048))
    model.add(Activation("relu"))
    model.add(Reshape(out_shape))
    model.summary()

    ##########################################

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(attribute_dim,))
    model_input = Concatenate()([noise, label])
    img = model(model_input)

    return Model([noise, label], img)