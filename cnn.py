from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from model import createModel

import numpy as np

import time
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

parser = argparse.ArgumentParser(description="BRATS CNN")
parser.add_argument('data_dir', metavar="data directory", type=str, help="The directory in which the training data is saved")
parser.add_argument('save_dir', metavar="model destination directory", type=str, help="The directory in which the model will be saved after training")
parser.add_argument('--model', metavar="model file", type=str, nargs='?', help="The hd5 file of the model to load")

args = parser.parse_args()
(images, labels, image_dimensions) = loadImages(args.data_dir)

samples = 100#183396

dataExtractor = DataExtractor(images, labels, image_dimensions)

X, y = dataExtractor.extractTrainingData(numSamples = samples)

print("Input data shape", X.shape, y.shape)

#add if statement checking if weigths for the model have been saved
if args.model == None:
    print("Creating a new model")
    model = createModel(X[0].shape)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    model_file = args.model
    print("Loading model from", model_file)
    model = load_model(model_file)

batch_size = 32
nb_epoch = 1
verbose = 1

model.fit(
        X, 
        y,
        batch_size = 32,
        nb_epoch = nb_epoch,
        verbose = verbose,
        validation_split = 0.1)

#datagen = ImageDataGenerator(rotation_range=20)
#
#model.fit_generator(
#        datagen.flow(X, y, batch_size=batch_size),
#        samples_per_epoch = X.shape[0],
#        nb_epoch = nb_epoch, 
#        verbose = verbose, 
#        callbacks = [checkpointer])


#Save the model
timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = args.save_dir + timestamp + ".h5"
model.save(filePath)
print("Saved the model to", filePath)
model.save('../models/latest.h5')
