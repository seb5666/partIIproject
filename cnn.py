from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from model import createModel
from inputParser import parse_input

import numpy as np
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

find_all_samples = True
use_N4Correction = True

args = parse_input()
data_dir = args.data_dir
validation_dir = args.validation_dir
model_file = args.model
save_dir = args.save_dir


(images, labels, image_dimensions) = loadImages(data_dir, use_N4Correction = use_N4Correction)
#Todo add N4 correction when images are ready
(val_images, val_labels, val_dimensions) = loadImages(validation_dir, use_N4Correction = use_N4Correction)

assert(image_dimensions == [image.shape for image in images])
assert(val_dimensions == [image.shape for image in val_images])

dataExtractor = DataExtractor(images, labels, val_images, val_labels, find_all_samples=find_all_samples, tf_ordering=tf_ordering)
X_train, y_train, X_val, y_val = dataExtractor.extractTrainingData()

print("Input data shape", X_train.shape, y_train.shape)
print("Validation data shape", X_val.shape, y_val.shape)

#add if statement checking if weigths for the model have been saved
if model_file == None:
    print("Creating a new model")
    model = createModel(X_train[0].shape, tf_ordering)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print("Loading model from", model_file)
    model = load_model(model_file)

model.summary()    

batch_size = 128
nb_epoch = 1
verbose = 1

rotateImages = True

print("Batch size", batch_size, "rotating images", rotateImages)

def rotate(k, Xs):
    if k == 0:
        return Xs
    else:
        rotated = []
        if tf_ordering: 
            rotated = [np.rot90(X, k=k) for X in Xs]
        else:
            rotated = [np.rot90(X, k=k, axes=(1,2)) for X in Xs]
        return np.array(rotated)

#rotate all images by all 4 possible 90 deg angles (0, 90, 180, 270)
def dataGenerator(X, y):
    datagen = ImageDataGenerator()
    for Xs, ys in datagen.flow(X, y, batch_size=batch_size):
        for i in range(4):
            yield rotate(i, Xs), ys

#train the model
if rotateImages:
    model.fit_generator(
        dataGenerator(X_train, y_train),
        samples_per_epoch = X_train.shape[0] * 4,
        nb_epoch = nb_epoch, 
        validation_data = dataGenerator(X_val, y_val),
        nb_val_samples = X_val.shape[0] * 4,
        verbose = verbose
        )
else:
    model.fit(
        X_train, 
        y_train,
        validation_data = (X_val, y_val),
        batch_size = batch_size,
        nb_epoch = nb_epoch,
        verbose = verbose
        )

#Save the model
timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + timestamp + ".h5"
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
