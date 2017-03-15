from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l1_l2
from keras.optimizers import SGD, Adam
import keras

def createModel(input_shape, tf_ordering=True, second_phase = False):
    print("Creating new model with input shape", input_shape)

    axis = -1
    if not(tf_ordering):
        axis = 1
    alpha = 0.1
    w_reg = 0.0001
    
    print("Hyperparameters: alpha=%f, w_reg=%f"%(alpha, w_reg))
    
    inputs = Input(shape=input_shape)

    p1 = Conv2D(filters=64, kernel_size=(7,7), padding='valid', kernel_regularizer=l1_l2(l1=w_reg, l2=w_reg), trainable=not(second_phase))(inputs)
    p1 = Dropout(alpha)(p1)
    p1 = Activation('relu')(p1)
    p1 = MaxPooling2D(pool_size=(4,4), strides=(1,1), border_mode='valid')(p1)
    
    p1 = Conv2D(filters=64, kernel_size=(3,3), padding='valid', kernel_regularizer=l1_l2(l1=w_reg, l2=w_reg), trainable=not(second_phase))(p1)
    p1 = Dropout(alpha)(p1)
    p1 = Activation('relu')(p1)
    p1 = MaxPooling2D(pool_size=(2,2), strides=(1,1), border_mode='valid')(p1)
   
    p2 = Conv2D(filters=160, kernel_size=(13,13), padding='valid', kernel_regularizer=l1_l2(l1=w_reg, l2=w_reg), trainable=not(second_phase))(inputs)
    p2 = Dropout(alpha)(p2)
    p2 = Activation('relu')(p2)
    
    merged = keras.layers.concatenate([p1, p2], axis=axis)

    predictions = Conv2D(filters=5, kernel_size=(21,21), padding='valid', kernel_regularizer=l1_l2(l1=w_reg, l2=w_reg))(merged)
    predictions = Dropout(alpha)(predictions)
    predictions = Flatten()(predictions)
    predictions = Activation('softmax')(predictions)
   
    model = Model(inputs=inputs, outputs=predictions)

    sgd = SGD(lr=0.005, decay = 1e-1, momentum=0.5, nesterov=True)
    adam = Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
