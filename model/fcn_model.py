from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Permute, Reshape, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

import numpy as np

def createModel(input_shape, tf_ordering=True, second_training_phase = False):

    alpha = 0.333
    axis = -1
    if not(tf_ordering):
        axis = 1

    l = 0
    init = 'glorot_normal'
    p = 0.0
    trainable = True
    print("Creating new fully convolutional model with input shape", input_shape)
    print("Parameters l=%f, alpha=%f, init=%s, p=%f"%(l, alpha, init, p))
    
    inputs = Input(shape=input_shape)

    h1 = Convolution2D(64, 3, 3, border_mode='same', init=init, input_shape=input_shape, W_regularizer=l2(l))(inputs)
    h1 = BatchNormalization(axis=axis)(h1)
    h1 = Activation('relu')(h1)

    h2 = Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h1) 
    h2 = BatchNormalization(axis=axis)(h2)
    h2 = Activation('relu')(h2)

    h3 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(h2)
    
    h4 = Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h3)
    h4 = BatchNormalization(axis=axis)(h4)
    h4 = Activation('relu')(h4)
    
    h5 = Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h4)
    h5 = BatchNormalization(axis=axis)(h5)
    h5 = Activation('relu')(h5)

    h6 = Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h5)
    h6 = BatchNormalization(axis=axis)(h6)
    h6 = Activation('relu')(h6)

    h7 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(h6)

    h8 = Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h7)
    h8 = BatchNormalization(axis=axis)(h8)
    h8 = Activation('relu')(h8)
    
    h9 = Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h8)
    h9 = BatchNormalization(axis=axis)(h9)
    h9 = Activation('relu')(h9)

    h10 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(h9)

    h11 = Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h10)
    h11 = BatchNormalization(axis=axis)(h11)
    h11 = Activation('relu')(h11)

    h12 = Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h11)
    h12 = BatchNormalization(axis=axis)(h12)
    h12 = Activation('relu')(h12)
    
    h13 = Convolution2D(5, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(h12)
    
    predictions = Reshape((5, 64))(h13)
    predictions = Permute((2,1))(predictions)
    predictions = Activation('softmax')(predictions)
    
    model = Model(input=[inputs], output=predictions)

    sgd = SGD(lr = 3e-5, decay =0.0, momentum = 0.9, nesterov = True)
    adam = Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
