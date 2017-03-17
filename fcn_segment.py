import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import math

import keras
from keras.models import load_model

from loadImagesFromDisk import loadTestImage
import sys
import os
from os import listdir

from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import pad
from metrics import dice

from keras.utils import np_utils 

assert(len(sys.argv) >= 4)

model_path = sys.argv[1]
image_dir_path = sys.argv[2]
output_file = sys.argv[3]

print("Model: {}".format(model_path))
print("Image directory: {}".format(image_dir_path))
print("Output: {}".format(output_file))

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False

print("Image ordering: {}".format(keras.backend.image_dim_ordering()))

use_N4Correction = True
print("Using N4 correction: {}".format(use_N4Correction))

batch_size = 128

patch_size=(64, 64)
label_size=(8,8)

model = load_model(model_path)

image, image_dimension = loadTestImage(image_dir_path, use_N4Correction = use_N4Correction)
print("Image dimension", image_dimension)

def normalize_channel(channel):
    channel_max = np.max(channel)
    channel_min = np.min(channel)
    channel = (channel - channel_min) / (channel_max - channel_min)
    print("Transforming input to range [0,1]")
    std = np.std(channel)
    if std != 0:
            return (channel - np.mean(channel)) / std
    else:
        return channel

#print(image[40, 60:70, 60:70, 0])
print("normalize each sequence")
for channel_num in range(4):
    channel = image[:, :, :, channel_num]
    
    #Transforming input to range [0,1]
    channel_max = np.max(channel)
    channel_min = np.min(channel)
    channel = (channel - channel_min) / (channel_max - channel_min)

    std = np.std(channel)
    if std != 0:
        channel = (channel - np.mean(channel)) / std
    
    image[:, :, :, channel_num] = channel
#print(image[40, 60:70, 60:70, 0])


height_padding = int((patch_size[0]-label_size[0])/2)
width_padding = int((patch_size[1]-label_size[1])/2)

# As the dimensions of the image not necesseraly a multiple of the label_size are, we need to add some extra padding for the last row and column, since the label_patch won't fit entirely in the original image
image = pad(image, ((0,0), (height_padding,height_padding + label_size[0]),(width_padding, width_padding + label_size[0]), (0,0)), mode='constant')
print("New dimension", image.shape)

segmentation = []
for i in range(0, image_dimension[0]):
    #print(i, end='')
    slice = image[i]
#    slice = np.arange(180 * 158).reshape((180, 158))
#    slice = pad(slice, ((height_padding,height_padding),(width_padding, width_padding)), mode='constant')
    patches = []
    for i in range(0, image_dimension[1], label_size[0]):
        for j in range(0, image_dimension[2], label_size[0]):
            startRow = i
            endRow = startRow + patch_size[0]
            startCol = j
            endCol = startCol + patch_size[1]
            p = slice[startRow: endRow, startCol: endCol]
            patches.append(p)
    patches = np.array(patches)
    if not(tf_ordering):
        patches = np.transpose(patches, (0,3,2,1))

    #not sure if batch size matters
    predictions = []
    i = 0
    while i < patches.shape[0]:
        if i + batch_size < patches.shape[0]:
            p = patches[i: i + batch_size]
        else:
            p = patches[i:]
        #preds = model.predict_classes(p, batch_size = p.shape[0], verbose=0)
        preds_prob = model.predict(p, batch_size = p.shape[0], verbose=0)
        #print(preds_prob.shape)
        preds = np.array([np_utils.probas_to_classes(probas) for probas in preds_prob])
        #print(preds.shape)
        predictions.extend(preds)
        i += batch_size

    predictions = np.array(predictions)

    height_in_patches = math.ceil(image_dimension[1]/label_size[0])
    width_in_patches = math.ceil(image_dimension[2]/label_size[1])
    predictions = predictions.reshape((height_in_patches, width_in_patches, label_size[0], label_size[1]))
#    predictions = np.transpose(predictions, (0,1,3,2))
    predictions = np.concatenate(predictions, axis = 1)
    predictions = np.concatenate(predictions, axis = 1)

    #print("Predictions5", predictions.shape)

    #transform linear array back into image
    slice = predictions[:image_dimension[1], :image_dimension[2]]

    #print("Slice dimension", slice.shape)
    segmentation.append(slice)

segmentation = np.array(segmentation)
#save segmentation
image = sitk.GetImageFromArray(segmentation)
image = sitk.Cast(image, sitk.sitkUInt8)
sitk.WriteImage(image, output_file)
print("Saved image")

def showImage(image, sliceNumber=72):
    plt.subplot(221)
    plt.imshow(image[sliceNumber, :, :, 0])
    plt.subplot(222)
    plt.imshow(image[sliceNumber, :, :, 1])
    plt.subplot(223)
    plt.imshow(image[sliceNumber, :, :, 2])
    plt.subplot(224)
    plt.imshow(image[sliceNumber, :, :, 3])
    plt.show()

def showSegmentation(segmentation, sliceNumber=72):
    plt.imshow(segmentation[sliceNumber, :, :])
    plt.show()

