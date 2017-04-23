from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from model import createModel
from inputParser import parse_input
from metrics import dice 
import numpy as np
import time
from normalization import normalize_scans

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

use_N4Correction = False

training_samples = 3000

patch_size = (33,33)

batch_size = 128
verbose = 1

rotateImages = True

print("Using N4 correction", use_N4Correction)
args = parse_input()
data_dir = args.data_dir
validation_dir = args.validation_dir
model_file = args.model
save_dir = args.save_dir

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + timestamp + ".h5"

if tf_ordering:
    shape = (patch_size[0], patch_size[1], 4)
else:
    shape = (4, patch_size[0], patch_size[1])

if model_file is None:
    print("Creating new model")
    model = createModel(shape, tf_ordering)
else:
    print("Loading model from", model_file)
    model = load_model(model_file, custom_objects={'dice':dice})

model.summary()    
print("Trainable weights", model.trainable_weights)

(images, labels, image_dimensions) = loadImages(data_dir, use_N4Correction = use_N4Correction)
print("Normalizing each scan")
images = normalize_scans(images, num_channels = 4)

#(val_images, val_labels, val_dimensions) = loadImages(validation_dir, use_N4Correction = use_N4Correction)

assert(image_dimensions == [image.shape for image in images])

print("Loaded %d training images"%len(images))

dataExtractor = DataExtractor(images, labels, tf_ordering=tf_ordering, patch_size = patch_size)

samples_weights = [1,1,1,1,1]  
print("Using weights for data", samples_weights)


print("Batch size", batch_size, "rotating images", rotateImages)

checkpointer = ModelCheckpoint(filepath=save_dir + timestamp + "_best.h5", verbose=verbose, save_best_only=True)

def dataGenerator(X, y):
    datagen = ImageDataGenerator()
    for Xs, ys in datagen.flow(X, y, batch_size=batch_size):
        if tf_ordering:
            rotated = [np.rot90(X, k = np.random.randint(0,4)) for X in Xs]
        else:
            rotated = [np.rot90(X, k = np.random.randint(0,4), axes=(1,2)) for X in Xs]
        yield np.array(rotated), ys

nb_epochs = 20
start_rate = 3e-5
end_rate = 3e-7
X_val = None
y_val = None
history = None
for i in range(nb_epochs):
    gc.collect()
    print("Epoch {}/{}".format(i+1, nb_epochs))  
    X_train, y_train, X_val2, y_val2 = dataExtractor.extractTrainingData(training_samples = training_samples)

    if X_val is None:
        X_val = X_val2
        y_val = y_val2
    print("Input data shape", X_train.shape, y_train.shape)
    print("Validation data shape", X_val.shape, y_val.shape)
    
    lr = start_rate + i * ((end_rate - start_rate) / (nb_epochs-1))
    model.optimizer.lr.set_value(lr)
    print("lr: {}".format(model.optimizer.lr.get_value()))
    hist = model.fit_generator(
        dataGenerator(X_train, y_train),
        samples_per_epoch = X_train.shape[0],
        nb_epoch = 1, 
        validation_data = (X_val, y_val),
        callbacks=[checkpointer],
        verbose = verbose
        )
    if history is None:
        history = hist.history
    else:
        for k in history:
            history[k].extend(hist.history[k])
print(history)
#Save the model
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
