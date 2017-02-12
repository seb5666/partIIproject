from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
from keras.optimizers import SGD

def createModel(input_shape, tf_ordering=True):
    print("Creating new model with input shape", input_shape)

    axis = -1
    if not(tf_ordering):
        axis = 1
    
    w_reg=0.01
    n_filters=[64,128,128,128]
    k_dims = [7,5,5,3]
    activation = 'relu'
    
    model = Sequential()
    model.add(Convolution2D(n_filters[0], k_dims[0], k_dims[0], border_mode='valid', W_regularizer=l1l2(l1 = w_reg, l2 = w_reg), input_shape = input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(mode=0, axis=axis))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    model.add(Dropout(0.5))


    model.add(Convolution2D(n_filters[1], k_dims[1], k_dims[1], border_mode='valid', W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    model.add(Activation(activation))
    model.add(BatchNormalization(mode=0, axis=axis))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(n_filters[2], k_dims[2], k_dims[2], border_mode='valid', W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    model.add(Activation(activation))
    model.add(BatchNormalization(mode=0, axis=axis))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(5))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model
