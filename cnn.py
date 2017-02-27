from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from model import createModel
from inputParser import parse_input
from metrics import dice 
import numpy as np
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import gc

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

find_all_samples = False
use_N4Correction = True
second_training_phase = False

training_samples = 450000
validation_samples = 45000

patch_size = (33,33)

batch_size = 128
nb_epoch = 10
verbose = 1

rotateImages = True

print("Finding all samples", find_all_samples)
print("Using N4 correction", use_N4Correction)
print("Second training phase with random samples", second_training_phase)
args = parse_input()
data_dir = args.data_dir
validation_dir = args.validation_dir
model_file = args.model
save_dir = args.save_dir

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + timestamp + ".h5"

(images, labels, image_dimensions) = loadImages(data_dir, use_N4Correction = use_N4Correction)
(val_images, val_labels, val_dimensions) = loadImages(validation_dir, use_N4Correction = use_N4Correction)

assert(image_dimensions == [image.shape for image in images])
assert(val_dimensions == [image.shape for image in val_images])

print("Loaded %d training images"%len(images))
print("Loaded %d validation images"%len(val_images))

dataExtractor = DataExtractor(images, labels, val_images, val_labels, find_all_samples=find_all_samples, tf_ordering=tf_ordering)

samples_weights = [3,1,1,1,1]  
print("Using weights for data", samples_weights)

if tf_ordering:
    shape = (patch_size[0], patch_size[1], 4)
else:
    shape = (4, patch_size[0], patch_size[1])

if model_file is None:
    print("Creating new model")
    model = createModel(shape, tf_ordering)
else:
    if second_training_phase:
        print("Loading model and setting all but dense layers as non trainable", model_file)
        model = createModel(shape, tf_ordering, second_training_phase=True)
        model.load_weights(model_file, custom_objects={'dice':dice})
    else:
        print("Loading model from", model_file)
        model = load_model(model_file, custom_objects={'dice':dice})

model.summary()    
print("Trainable weights", model.trainable_weights)

print("Batch size", batch_size, "rotating images", rotateImages)

checkpointer = ModelCheckpoint(filepath=save_dir + timestamp + "_best.h5", verbose=verbose, save_best_only=True)

trainingDataGenerator = ImageDataGenerator(horizontal_flip=rotateImages, vertical_flip=rotateImages)

nb_epochs = 20
start_rate = 3e-5
end_rate = 3e-7
X_val = None
y_val = None
for i in range(nb_epochs):
    gc.collect()
    print("Epoch {}".format(i))  

    X_train, y_train, X_val2, y_val2 = dataExtractor.extractTrainingData(samples_weights = samples_weights, training_samples = training_samples, validation_samples=validation_samples, patch_size=patch_size)

    if X_val is None:
        X_val = X_val2
        y_val = y_val2
    print("Input data shape", X_train.shape, y_train.shape)
    print("Validation data shape", X_val.shape, y_val.shape)
    
    lr = start_rate + i * ((end_rate - start_rate) / (nb_epochs-1))
    print("lr: {}".format(lr))
    model.optimizer.lr.set_value(lr)
    print(model.optimizer.lr.get_value())
    model.fit_generator(
        trainingDataGenerator.flow(X_train, y_train, batch_size = batch_size),
        samples_per_epoch = X_train.shape[0],
        nb_epoch = 1, 
        validation_data = (X_val, y_val),
        callbacks=[checkpointer],
        verbose = verbose
        )

#Save the model
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
