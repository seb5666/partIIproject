from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

from metrics import per_class_precision, dice

def createModel(input_shape, tf_ordering=True, second_training_phase = False):

    alpha = 0.333
    axis = -1
    if not(tf_ordering):
        axis = 1

    l = 0.0001
    
    print("Creating new model with input shape", input_shape)
    print("Parameters l=%f, alpha=%f"%(l, alpha))

    trainable = not(second_training_phase)
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape, W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid', trainable=trainable))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha, trainable=trainable))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l), trainable=trainable))
    model.add(BatchNormalization(axis=axis, trainable=trainable))
    model.add(LeakyReLU(alpha))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid', trainable=trainable))

    model.add(Flatten())

    model.add(Dense(256, W_regularizer=l2(l)))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.1))

    model.add(Dense(256, W_regularizer=l2(l)))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.1))

    model.add(Dense(5, W_regularizer=l2(l)))
    model.add(BatchNormalization(axis=axis))
    model.add(Activation('softmax'))
   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', dice])

    return model
