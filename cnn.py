from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
from model import createModel
from inputParser import parse_input

import numpy as np
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

args = parse_input()
data_dir = args.data_dir
model_file = args.model
save_dir = args.save_dir

(images, labels, image_dimensions) = loadImages(data_dir)

#the smallest class consists of 183396 patches, thus we have 5 times that many samples in total
samples = 1000 #183396 * 5

dataExtractor = DataExtractor(images, labels, image_dimensions)
X, y = dataExtractor.extractTrainingData(numSamples = samples)
print("Input data shape", X.shape, y.shape)

#add if statement checking if weigths for the model have been saved
if model_file == None:
    print("Creating a new model")
    model = createModel(X[0].shape)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    print("Loading model from", model_file)
    model = load_model(model_file)
    
batch_size = 32
nb_epoch = 1
verbose = 1
rotateImages = True

def rotate(k, Xs):
    if k == 0:
        return Xs
    else:
        rotated = [np.rot90(X) for X in Xs]
        return np.array(rotated)

#rotate all images by all 4 possible 90 deg angles (0, 90, 180, 270)
def dataGenerator():
    datagen = ImageDataGenerator()
    for Xs, ys in datagen.flow(X, y, batch_size=batch_size):
        for i in range(4):
            yield rotate(i, Xs), ys

#train the model
if rotateImages:
    model.fit_generator(
        dataGenerator(),
        samples_per_epoch = X.shape[0] * 4,
        nb_epoch = nb_epoch, 
        verbose = verbose
        )
else:
    model.fit(
        X, 
        y,
        batch_size = 32,
        nb_epoch = nb_epoch,
        verbose = verbose
        )



#Save the model
timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + timestamp + ".h5"
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
