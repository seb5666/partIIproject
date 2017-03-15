from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from double_pathway_model import createModel
from inputParser import parse_input
from metrics import dice
import numpy as np 
import time
import gc

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
second_training_phase = True
patch_size = (33,33)

batch_size = 128
nb_epochs = 1
verbose = 1

training_samples = 1000
validation_samples = 100
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

dataExtractor = DataExtractor(images, labels, val_images, val_labels, find_all_samples=find_all_samples, tf_ordering=tf_ordering, patch_size = patch_size)
patches_per_class = [len(patches) for patches in dataExtractor.valid_training_patches]
print("VALID PATCHES PER CLASS", [p/sum(patches_per_class) for p in patches_per_class])
samples_weights = [1,1,1,1,1]  
print("Using weights for data", samples_weights)

if tf_ordering:
    shape = (patch_size[0], patch_size[1], 4)
else:
    shape = (4, patch_size[0], patch_size[1])

if model_file == None:
    print("Creating new model")
    model  = createModel(shape, tf_ordering, second_phase = second_training_phase)
else:
    if second_training_phase:
        print("Loading model from weights and setting all but last layer as non trainable", model_file)
        model = createModel(shape, tf_ordering, second_phase = True)
        model.load_weights(model_file)
    else:
        print("Loading model", model_file)
        model = load_model(model_file, custom_objects={'dice':dice})
        print("Setting momentum to 0.9")
        model.optimizer.momentum.set_value(0.9)
        print("Model momentum", model.optimizer.momentum.get_value())

model.summary()    
print("Trainable weights", model.trainable_weights)

print("Batch size", batch_size, "rotating images", rotateImages)

checkpointer = ModelCheckpoint(filepath=save_dir + timestamp + "_best.h5", verbose=verbose, save_best_only=True)

training_data_generator = ImageDataGenerator(horizontal_flip=rotateImages, vertical_flip=rotateImages)

X_val = None
y_val = None
for i in range(nb_epochs):
    gc.collect() #import to avoud memory errors..
    
    print("Epoch {}".format(i))
    
    if second_training_phase:
        X_train, y_train, X_val2, y_val2 = dataExtractor.extractRandomTrainingData(training_samples=training_samples, validation_samples = validation_samples)
    else:
        X_train, y_train, X_val2, y_val2 = dataExtractor.extractTrainingData(samples_weights = samples_weights, training_samples = training_samples, validation_samples = validation_samples)

    if X_val is None:
        X_val = X_val2
        y_val = y_val2

    model.fit_generator(
        training_data_generator.flow(X_train, y_train, batch_size = batch_size),
        samples_per_epoch = X_train.shape[0],
        epochs = 1, 
        validation_data = (X_val, y_val),
        callbacks=[checkpointer],
        verbose = verbose
        )

#Save the model
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
