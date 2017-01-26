from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def createModel(input_shape):
    
    LReLu = LeakyReLU(alpha=0.333)

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LReLu)

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LReLu)
    #model.add(Dropout(0.1))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LReLu)
    #model.add(Dropout(0.1))

    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model
