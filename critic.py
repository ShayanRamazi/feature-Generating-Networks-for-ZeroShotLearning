#implement Critic

#######################################################################

from __future__ import print_function, division
from keras.layers import Input, Dense, Concatenate,LeakyReLU
from keras.models import Sequential, Model

#######################################################################

def build_critic():

    ##########################################
    attribute_dim = 85
    features_dim  = 2048
    in_shape      = attribute_dim + features_dim
    ##########################################

    model = Sequential()
    model.add(Dense(2048, input_dim=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))
    model.summary()

    ##########################################

    feature = Input(shape=(features_dim,))
    label = Input(shape=(attribute_dim,))
    model_input = Concatenate()([feature, label])
    validity = model(model_input)

    return Model([feature, label], validity)