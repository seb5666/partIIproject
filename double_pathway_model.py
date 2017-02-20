from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1l2
from keras.engine.topology import Merge

def createModel(input_shape, tf_ordering=True):
    print("Creating new model with input shape", input_shape)

    axis = -1
    if not(tf_ordering):
        axis = 1

    w_reg = 0.0001
    
    path1 = Sequential()
    path1.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape = input_shape, W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    path1.add(BatchNormalization(axis=axis))
    path1.add(Activation('relu'))
    path1.add(MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid'))

    path1.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    path1.add(BatchNormalization(axis=axis))
    path1.add(Activation('relu'))
    path1.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid'))
    print(path1.input_shape)
    print(path1.output_shape)
   
    path2 = Sequential()
    path2.add(Convolution2D(160, 13, 13, border_mode='valid', input_shape = input_shape, W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    path2.add(BatchNormalization(axis=axis))
    path2.add(Activation('relu'))
    print(path2.input_shape)
    print(path2.output_shape)

    merge_layer = Sequential()
    merge_layer.add(Merge([path1, path2], mode='concat', concat_axis=axis))
    print(merge_layer.input_shape)

    classification_layer = Sequential()
    classification_layer.add(merge_layer)
    print(classification_layer.output_shape)
    classification_layer.add(Convolution2D(5, 21, 21, border_mode='valid', W_regularizer=l1l2(l1 = w_reg, l2 = w_reg)))
    classification_layer.add(BatchNormalization(axis=axis))
    print(classification_layer.output_shape)
    classification_layer.add(Flatten())
    classification_layer.add(Activation('softmax'))
   
    classification_layer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    path1.summary()
    path2.summary()
    classification_layer.summary()
    
    return classification_layer
