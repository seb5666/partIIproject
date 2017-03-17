from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout, Permute, Reshape, Input, Merge, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
import keras
import numpy as np

from keras.utils.visualize_util import plot

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
    
    inputs1 = Input(shape=input_shape[0])
    inputs2 = Input(shape=input_shape[1])
    
    path1 =  Convolution2D(20, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(inputs1)
    path1 = BatchNormalization(axis=axis)(path1)
    path1 = Activation('relu')(path1)

    path1 =  Convolution2D(20, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path1)
    path1 = BatchNormalization(axis=axis)(path1)
    path1 = Activation('relu')(path1)

    path1 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(path1)
    
    path1 =  Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path1)
    path1 = BatchNormalization(axis=axis)(path1)
    path1 = Activation('relu')(path1)
    
    path1 =  Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path1)
    path1 = BatchNormalization(axis=axis)(path1)
    path1 = Activation('relu')(path1)
    
    path1 =  Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path1)
    path1 = BatchNormalization(axis=axis)(path1)
    path1 = Activation('relu')(path1)
    
    path2 = Convolution2D(20, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(inputs2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)

    path2 = Convolution2D(20, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)
   
    path2 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(path2)
    
    path2 = Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)
    
    path2 = Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)
    
    path2 = Convolution2D(40, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)

    path2 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(path2)
    
    path2 = Convolution2D(60, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)
    
    path2 = Convolution2D(60, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)
    
    path2 = Convolution2D(60, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(path2)
    path2 = BatchNormalization(axis=axis)(path2)
    path2 = Activation('relu')(path2)

    path2 = MaxPooling2D(pool_size=(2,2), border_mode='valid')(path2)
    
    merged = merge([path1, path2], mode='concat', concat_axis=axis)

    predictions = Convolution2D(80, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(merged)
    predictions = BatchNormalization(axis=axis)(predictions)
    predictions = Activation('relu')(predictions)

    predicionts = Convolution2D(80, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(predictions)
    predictions = BatchNormalization(axis=axis)(predictions)
    predictions = Activation('relu')(predictions)

    predictions = MaxPooling2D(pool_size=(2,2), border_mode='valid')(predictions)

    predicionts = Convolution2D(100, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(predictions)
    predictions = BatchNormalization(axis=axis)(predictions)
    predictions = Activation('relu')(predictions)
    
    predicionts = Convolution2D(100, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(predictions)
    predictions = BatchNormalization(axis=axis)(predictions)
    predictions = Activation('relu')(predictions)

    
    predictions = Convolution2D(5, 3, 3, border_mode='same', init=init, W_regularizer=l2(l))(predictions)
    
    predictions = Reshape((5, 16))(predictions)
    predictions = Permute((2,1))(predictions)
    predictions = Activation('softmax')(predictions)
    
    model = Model(input=[inputs1, inputs2], output=predictions)

    sgd = SGD(lr = 3e-5, decay =0.0, momentum = 0.9, nesterov = True)
    adam = Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    plot(model, to_file='model.png')
    return model
