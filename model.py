from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from metrics import per_class_precision, dice

import numpy as np

def createModel(input_shape, tf_ordering=True, second_training_phase = False):

    alpha = 0.333
    axis = -1
    if not(tf_ordering):
        axis = 1

    l = 0
    init = 'glorot_normal'

    print("Creating new model with input shape", input_shape)
    print("Parameters l=%f, alpha=%f, init=%s"%(l, alpha, init))

    trainable = not(second_training_phase)
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, input_shape = input_shape, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid', trainable=trainable))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha, trainable=trainable))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l), trainable=trainable))
    #model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid', trainable=trainable))

    model.add(Flatten())

    model.add(Dense(256, init=init, W_regularizer=l2(l)))
    #model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.1))

    model.add(Dense(256, init=init, W_regularizer=l2(l)))
    #model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.1))

    model.add(Dense(5, init=init, W_regularizer=l2(l)))
    #model.add(BatchNormalization(axis=axis))
    model.add(Activation('softmax'))
    
    #Set bias weights to 0.1 
    b = 0.1
    print("Setting the bias weights to", b)
    weights = model.get_weights()
    for i in range(1, len(weights) - 2, 2):
        print(weights[i].shape)
        weights[i] = np.full(weights[i].shape, b)
    model.set_weights(weights)

    sgd = SGD(lr = 3e-5, decay =0.0, momentum = 0.9, nesterov = True)
    adam = Adam()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', dice])
    
    return model
