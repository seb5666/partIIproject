from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D

def createModel(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    return model
