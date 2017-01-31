from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def createModel(input_shape):
    print("Creating new model with input shape", input_shape)

    alpha = 0.333

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))

    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.1))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    #model.add(Dropout(0.1))

    model.add(Dense(5))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    return model
