from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from double_pathway_model import createModel
from inputParser import parse_input
import numpy as np
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

find_all_samples = False
use_N4Correction = True
second_training_phase = False

batch_size = 128
nb_epoch = 1
verbose = 1

training_samples = 4500
validation_samples = 450
rotateImages = True

print("Finding all samples", find_all_samples)
print("Using N4 correction", use_N4Correction)
print("Second training phase", second_training_phase)

args = parse_input()
data_dir = args.data_dir
validation_dir = args.validation_dir
model_file = args.model
save_dir = args.save_dir

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + 'double_pathway' + timestamp + ".h5"

(images, labels, image_dimensions) = loadImages(data_dir, use_N4Correction = use_N4Correction)
(val_images, val_labels, val_dimensions) = loadImages(validation_dir, use_N4Correction = use_N4Correction)

assert(image_dimensions == [image.shape for image in images])
assert(val_dimensions == [image.shape for image in val_images])

print("Loaded %d training images"%len(images))
print("Loaded %d validation images"%len(val_images))

dataExtractor = DataExtractor(images, labels, val_images, val_labels, find_all_samples=find_all_samples, tf_ordering=tf_ordering)
if not(second_training_phase):
    samples_weights = [1,1,1,1,1]  
    X_train, y_train, X_val, y_val = dataExtractor.extractTrainingData(samples_weights = samples_weights)
    print("Using weights for data", samples_weights)
else:
    training_samples = 450000
    validation_samples = 45000
    X_train, y_train, X_val, y_val = dataExtractor.extractRandomTrainingData(training_samples = training_samples, validation_samples = validation_samples)

print("Input data shape", X_train.shape, y_train.shape)
print("Validation data shape", X_val.shape, y_val.shape)

if model_file == None:
    print("Creating new model")
    model  = createModel(X_train[0].shape, tf_ordering)
else:
    if second_training_phase:
        print("Loading model from weights and setting all but last layer as non trainable", model_file)
        model = createModel(X_train[0].shape, tf_ordering, second_phase = True)
        model.load_weights(model_file)
    else:
        print("Loading model", model_file)
        model = load_model(model_file)

model.summary()    
print("Trainable weights", model.trainable_weights)

print("Batch size", batch_size, "rotating images", rotateImages)

checkpointer = ModelCheckpoint(filepath=save_dir + timestamp + "_best.h5", verbose=0, save_best_only=True)

trainingDataGenerator = ImageDataGenerator(horizontal_flip=rotateImages, vertical_flip=rotateImages)

def two_pathway_generator(X_train, y_train, batch_size):
    for (X, y) in trainingDataGenerator.flow(X_train, y_train, batch_size):
        yield [X, X], y

model.fit_generator(
    two_pathway_generator(X_train, y_train, batch_size = batch_size),
    samples_per_epoch = X_train.shape[0],
    nb_epoch = nb_epoch, 
    validation_data = ([X_val, X_val], y_val),
    callbacks=[checkpointer],
    verbose = verbose
    )

#Save the model
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
