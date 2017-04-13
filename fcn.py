from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from fcn_model import createModel
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

args = parse_input()
data_dir = args.data_dir
validation_dir = args.validation_dir
model_file = args.model
save_dir = args.save_dir

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

use_N4Correction = True
print("Using N4 correction", use_N4Correction)

training_samples = 100000
#validation_samples = 20000

patch_size = (64,64)
label_size = (8,8)

print("Patch size {}, label size: {}".format(patch_size, label_size))
batch_size = 128
nb_epochs = 40
verbose = 1

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S')
filePath = save_dir + timestamp + ".h5"

num_channels = 4
if tf_ordering:
    shape = (patch_size[0], patch_size[1], num_channels)
else:
    shape = (num_channels, patch_size[0], patch_size[1])

if model_file is None:
    print("Creating new model")
    model = createModel(shape, tf_ordering)
else:
    print("Loading model from", model_file)
    model = load_model(model_file)

model.summary()
print("Trainable weights", model.trainable_weights)

(images, labels, image_dimensions) = loadImages(data_dir, use_N4Correction = use_N4Correction)
#(val_images, val_labels, val_dimensions) = loadImages(validation_dir, use_N4Correction = use_N4Correction)

assert(image_dimensions == [image.shape for image in images])
#assert(val_dimensions == [image.shape for image in val_images])
print("Loaded %d training images"%len(images))
#print("Loaded %d validation images"%len(val_images))

dataExtractor = DataExtractor(images, labels, [], [], tf_ordering=tf_ordering, patch_size = patch_size, label_size=label_size)

print("Batch size", batch_size)

checkpointer = ModelCheckpoint(filepath=save_dir + timestamp + "_best.h5", verbose=verbose, save_best_only=True)

trainingDataGenerator = ImageDataGenerator()

start_rate = 3e-3
end_rate = 3e-5

to_categorical = np.vectorize(np_utils.to_categorical)

X_val = None
y_val = None
history = None
for i in range(nb_epochs):
    gc.collect()
    print("Epoch {}/{}".format(i+1, nb_epochs))  

    X_train, y_train, X_val2, y_val2 = dataExtractor.extractLargeTrainingData(training_samples = training_samples)
        
    if X_val is None:
        X_val = X_val2
        y_val = y_val2.reshape((-1,label_size[0] * label_size[1], 5))

    print("Input data shape", X_train.shape, y_train.shape)
    print("Validation data shape", X_val.shape, y_val.shape)
    
    #lr = start_rate + i * ((end_rate - start_rate) / (nb_epochs-1))
    #model.optimizer.lr.set_value(lr)
    #print("lr: {}".format(model.optimizer.lr.get_value()))
    hist = model.fit_generator(
        trainingDataGenerator.flow(X_train, y_train, batch_size = batch_size),
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
    model.save(filePath[:-3] + "epoch_" + str(i) + ".h5")
print(history)

#Save the model
model.save(filePath)
print("Saved the model to", filePath)
model.save(save_dir + 'latest.h5')
