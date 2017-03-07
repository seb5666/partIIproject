from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Permute, Reshape
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
    p = 0.1
    trainable = True
    print("Creating new fully convolutional model with input shape", input_shape)
    print("Parameters l=%f, alpha=%f, init=%s, p=%f"%(l, alpha, init, p))

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, input_shape = input_shape, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))


    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid', trainable=trainable))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid', trainable=trainable))

    model.add(Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(Dropout(p))
    model.add(LeakyReLU(alpha))
    
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid', trainable=trainable))

    model.add(Convolution2D(5, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    
    model.add(Reshape((5, 64)))
    model.add(Permute((2,1)))
    model.add(Activation('softmax'))
    sgd = SGD(lr = 3e-5, decay =0.0, momentum = 0.9, nesterov = True)
    adam = Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
