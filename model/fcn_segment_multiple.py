import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import math

from sklearn.feature_extraction.image import extract_patches_2d
import skimage

from keras.utils import np_utils 

from fcn_segment import segment

if __name__ == '__main__':

    import keras
    from keras.models import load_model

    import sys
    import os
    from os import listdir

    from loadImagesFromDisk import loadTestImage
    from normalization import normalize_scans
    
    import time
    start_time = time.time()
    
    assert(len(sys.argv) >= 4)
    model_path = sys.argv[1]
    images_dir_path = sys.argv[2]
    output_dir = sys.argv[3]

    print("Model: {}".format(model_path))
    print("Images directory: {}".format(images_dir_path))
    print("Output dir: {}".format(output_dir))

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
    
    for index, dir in enumerate(listdir(images_dir_path)):
        if "brats" in dir:
            image_dir_path = os.path.join(images_dir_path, dir)
            print("Segmenting image {}".format(image_dir_path))
            image, image_dimension = loadTestImage(image_dir_path, use_N4Correction = use_N4Correction)
            image = normalize_scans([image], num_channels = 4)[0]

            print("Image dimension", image_dimension)

            segmentation = segment(image, image_dimension, patch_size, label_size, model.predict, batch_size = batch_size, tf_ordering=tf_ordering)

            #save segmentation
            segmentation = sitk.GetImageFromArray(segmentation)
            segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)
            output_file = os.path.join(output_dir, dir) + ".mha"
            sitk.WriteImage(segmentation, output_file)
            print("Saved image to {}".format(output_file))

    print("--- {} seconds ---".format(time.time() - start_time))
