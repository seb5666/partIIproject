import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import keras
from keras.models import load_model

from loadImagesFromDisk import loadTestImage
import sys
import os
from os import listdir

from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import pad
from metrics import dice

assert(len(sys.argv) >= 4)

tf_ordering = True
if (keras.backend.image_dim_ordering() == "th"):
    tf_ordering = False
print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

double_path_architecture = True

model_path = sys.argv[1]
image_dir_path = sys.argv[2]
output_file = sys.argv[3] 
print("Output: ", output_file)

model = load_model(model_path, custom_objects={'dice': dice})

normalize_channels = True
normalize_per_patch = False
normalize_with_training_values = False
means = [157.3013, 89.12162, 99.944237, 62.016068]
stds = [360.25491, 232.29645, 216.15245, 138.30151]

batch_size = 128
patch_size=(33,33)


image, image_dimension = loadTestImage(image_dir_path, use_N4Correction = True)
print("Image dimension", image_dimension)

def normalize_patches(patches):
    if tf_ordering:
        for p_index, patch in enumerate(patches):
            for channel in range(patch.shape[-1]):
                patches[p_index, :, :, channel] = normalize_channel(patch[:, :, channel])
        else:
            for p_index, patch in enumerate(patches):
                for channel in range(patch.shape[0]):
                    patches[p_index, channel, :, :] = normalize_channel(patch[channel, :, :])
    return patches

def normalize_channel(channel):
    std = np.std(channel)
    if std != 0:
            return (channel - np.mean(channel)) / std
    else:
        return channel

if normalize_channels:
    #print(image[70, 80:100, 80:100, 0])
    #normalize each channel
    if not normalize_per_patch:
        for channel in range(4):
            image[:, :, :, channel] = normalize_channel(image[:, :, :, channel])
    #print(image[70, 80:100, 80:100, 0])

half_height = int(patch_size[0]/2)
half_width = int(patch_size[1]/2)

image = pad(image, ((0,0), (half_height,half_height),(half_width, half_width), (0,0)), mode='constant')
print("New dimension", image.shape)

segmentation = []
for i in tqdm(range(0, image_dimension[0])):
    #print("Slice", i)
    patches = extract_patches_2d(image[i], patch_size)
    
    if normalize_per_patch:
        patches = normalize_patches(patches)

    if normalize_with_training_values:
        print("Normalizing with training patches means and stds")
        num_channels = 4
        epsilon = 0.0001
   
        means = [418.99301, 424.04623, 428.94061, 433.69357]
        stds = [396.99472, 397.67438, 398.29724, 398.90411]

        print("Training patches means on training data", means)
        print("Training patches std on training data", stds)
    
        print("Normalising each patch")
        for patch in pathches:
            for c in range(num_channels):
                patch[:,:,c] = (patch[:,:,c] - means[c]) / stds[c]
        print("Done normalizing")
    
    if not(tf_ordering):
        patches = np.transpose(patches, (0,3,2,1))
    
    #print("Patches", patches.shape)

    #not sure if batch size matters
    predictions = np.empty((0,))
    i = 0
    while i < patches.shape[0]:
        if i + batch_size < patches.shape[0]:
            p = patches[i: i + batch_size]
        else:
            p = patches[i:]
        if double_path_architecture:
            preds = model.predict_classes([p,p], batch_size = p.shape[0], verbose=0)
        else:
            preds = model.predict_classes(p, batch_size = p.shape[0], verbose=0)
        predictions = np.concatenate((predictions, preds))
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
image = sitk.Cast(image, sitk.sitkUInt8)
sitk.WriteImage(image, output_file)

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
