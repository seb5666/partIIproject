from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

def createModel(input_shape, tf_ordering=True):
    print("Creating new model with input shape", input_shape)

    alpha = 0.333
    axis = -1
    if not(tf_ordering):
        axis = 1

    l = 0.0001

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape, W_regularizer=l2(l)))
    #model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(l)))
    #model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(l)))
    #model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l)))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l)))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same', W_regularizer=l2(l)))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
    
    model.add(Convolution2D(256, 3, 3, border_mode='valid', W_regularizer=l2(l)))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))


    model.add(Convolution2D(256, 3, 3, border_mode='valid', W_regularizer=l2(l)))
    #model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization(axis=axis))
    model.add(LeakyReLU(alpha))

    model.add(Flatten())

    #model.add(Dense(256, W_regularizer=l2(l)))
    #model.add(Dense(256))
    #model.add(BatchNormalization(axis=axis))
    #model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.1))

    #model.add(Dense(256, W_regularizer=l2(l)))
    #model.add(Dense(256))
    #model.add(BatchNormalization(axis=axis))
    #model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.1))

    model.add(Dense(5, W_regularizer=l2(l)))
    #model.add(Dense(5))
    model.add(BatchNormalization(axis=axis))
    model.add(Activation('softmax'))
   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
