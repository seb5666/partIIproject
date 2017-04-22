import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import math

from sklearn.feature_extraction.image import extract_patches_2d
import skimage

from keras.utils import np_utils 

def pad(image, patch_size, label_size):
    height_padding = int((patch_size[0]-label_size[0])/2)
    width_padding = int((patch_size[1]-label_size[1])/2)

    # As the dimensions of the image not necesseraly a multiple of the label_size are, we need to add some extra padding for the last row and column, since the label_patch won't fit entirely in the original image
    return skimage.util.pad(image, ((0,0), (height_padding,height_padding + label_size[0]),(width_padding, width_padding + label_size[0]), (0,0)), mode='constant')

def extract_patches(slice, image_dimension, patch_size, label_size, tf_ordering = True):
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
    return patches

def segment(image, image_dimension, patch_size, label_size, classify, batch_size = 1, verbose=True, tf_ordering = True):
    padded_image = pad(image, patch_size, label_size)

    height_in_patches = math.ceil(image_dimension[1]/label_size[0])
    width_in_patches = math.ceil(image_dimension[2]/label_size[1])

    segmentation = []
    for i in tqdm(range(0, image_dimension[0]), disable = not(verbose)):
        slice = padded_image[i]
        
        patches = extract_patches(slice, image_dimension, patch_size, label_size, tf_ordering)

        predictions = []
        i = 0
        while i < patches.shape[0]:
            if i + batch_size < patches.shape[0]:
                p = patches[i: i + batch_size]
            else:
                p = patches[i:]
            preds_prob = classify(p, batch_size = p.shape[0], verbose=0)
            preds = np.array([np_utils.probas_to_classes(probas) for probas in preds_prob])
            predictions.extend(preds)
            i += batch_size

        predictions = np.array(predictions)

        predictions = predictions.reshape((height_in_patches, width_in_patches, label_size[0], label_size[1]))
        predictions = np.concatenate(predictions, axis = 1)
        predictions = np.concatenate(predictions, axis = 1)

        #transform linear array back into image
        slice = predictions[:image_dimension[1], :image_dimension[2]]

        segmentation.append(slice)
    return np.array(segmentation)

if __name__ == '__main__':

    import keras
    from keras.models import load_model

    import sys
    import os
    from os import listdir

    from loadImagesFromDisk import loadTestImage
    from normalization import normalize_scans
    
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
    image = normalize_scans([image], num_channels = 4)[0]

    print("Image dimension", image_dimension)

    segmentation = segment(image, image_dimension, patch_size, label_size, model.predict, batch_size = batch_size, tf_ordering=tf_ordering)

    #save segmentation
    segmentation = sitk.GetImageFromArray(segmentation)
    segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)
    sitk.WriteImage(image, output_file)
    print("Saved image")

