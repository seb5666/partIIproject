from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

(slices, labels, imageDimension) = loadImages('../Dataset/BRATS-2/Synthetic_Data/HG/')

print(slices.shape)
print(labels.shape)
print(imageDimension)

samples = 100

dataExtractor = DataExtractor(slices, labels, imageDimension)
X, y = dataExtractor.extractTrainingData(numSamples = samples)

print(X.shape)
print(y.shape)

print("Model shapes:")
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
print(model.output_shape)

model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Convolution2D(128, 3, 3, border_mode='same', input_shape=X[0].shape))
model.add(Activation('relu'))
print(model.output_shape)

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='valid'))
model.add(Flatten())
print(model.output_shape)

model.add(Dense(256))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Dense(256))
model.add(Activation('relu'))
print(model.output_shape)

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(optimizer='adadelta', 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

batch_size = 32
nb_epoch = 10
verbose = 1

checkpointer = ModelCheckpoint('./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5')

model.fit(
        X, 
        y,
        batch_size = 32,
        nb_epoch = nb_epoch,
        verbose = verbose,
        callbacks = [checkpointer],
        validation_split = 0.1)

#datagen = ImageDataGenerator(rotation_range=20)

#model.fit_generator(
#        datagen.flow(X, y, batch_size=batch_size),
#        samples_per_epoch = X.shape[0],
#        nb_epoch = nb_epoch, 
#        verbose = verbose, 
#        callbacks = [checkpointer])
