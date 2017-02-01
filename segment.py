import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import load_model

from loadImagesFromDisk import loadTestImage
import sys
import os
from os import listdir

from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import pad

assert(len(sys.argv) >= 3)

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

model_path = sys.argv[1]
image_dir_path = sys.argv[2]

model = load_model(model_path)

batch_size = 128
patch_size=(33,33)

image, image_dimension = loadTestImage(image_dir_path, use_N4Correction = False)
print("Image dimension", image_dimension)

# normalize images
def normalize(slice):
    if np.std(slice) == 0:
        return slice
    else:
        return (slice - np.mean(slice)) / np.std(slice)
for slice in image:
    slice = normalize(slice)

#extract patches + padd for edges?
half_height= int(patch_size[0]/2)
half_width = int(patch_size[1]/2)

image = pad(image, ((0,0), (half_height,half_height),(half_width, half_width), (0,0)), mode='constant')
print("New dimension", image.shape)

segmentation = []
for i in range(0, image_dimension[0]):
    print("Slice", i)
    patches = extract_patches_2d(image[i], patch_size)
    if not(tf_ordering):
        patches = np.transpose(patches, (0,3,2,1))
    
    print("Patches", patches.shape)

    #not sure if batch size matters
    predictions = []
    i = 0
    while i < patches.shape[0]:
        if i + batch_size < patches.shape[0]:
            p = patches[i: i + batch_size]
        else:
            p = patches[i:]
        predictions.append(model.predict_classes(p, batch_size = p.shape[0], verbose=0))
        i += batch_size
    predictions = np.array(predictions)

    #print("Predictions", predictions.shape)

    #transform linear array back into image
    slice = predictions.reshape(image_dimension[1], image_dimension[2])

    #print("Slice dimension", slice.shape)
    #plt.imshow(slice)
    #plt.show()

    segmentation.append(slice)

segmentation = np.array(segmentation)

#save segmentation
image = sitk.GetImageFromArray(segmentation)
sitk.WriteImage(image, "output.mha")

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
