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
equiprobable_classes = True

print("Finding all samples", find_all_samples)
print("Using N4 correction", use_N4Correction)
print("Using equiprobable classes for training data", equiprobable_classes)

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
if equiprobable_classes:
    samples_weights = [1,1,1,1,1]  
    X_train, y_train, X_val, y_val = dataExtractor.extractTrainingData(samples_weights = samples_weights)
    print("Using weights for data", samples_weights)
else:
    X_train, y_train, X_val, y_val = dataExtractor.extractRandomTrainingData()
print("Input data shape", X_train.shape, y_train.shape)
print("Validation data shape", X_val.shape, y_val.shape)

#add if statement checking if weigths for the model have been saved
if model_file == None:
    print("Creating a new model")
    model = createModel(X_train[0].shape, tf_ordering)
else:
    print("Loading model from", model_file)
    model = load_model(model_file)

model.summary()    

batch_size = 128
nb_epoch = 1
verbose = 1

rotateImages = True

print("Batch size", batch_size, "rotating images", rotateImages)

trainingDataGenerator = ImageDataGenerator(horizontal_flip=True, vertical_flip=False)

#train the model
if rotateImages:
    model.fit_generator(
        trainingDataGenerator.flow(X_train, y_train, batch_size = batch_size),
        samples_per_epoch = X_train.shape[0],
        nb_epoch = nb_epoch, 
        validation_data = (X_val, y_val),
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
