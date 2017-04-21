import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.image import extract_patches_2d
import skimage

def normalize(image):
    for channel in range(4):
        modality = image[:, :, :, channel]
        std = np.std(channel)
        if std != 0:
            image[:, :, :, channel] = (channel - np.mean(channel)) / std

def pad(image, patch_size):
    half_height = int(patch_size[0]/2)
    half_width = int(patch_size[1]/2)

    return skimage.util.pad(image, ((0,0), (half_height,half_height),(half_width, half_width), (0,0)), mode='constant')

def segment(image, patch_size, classify,  tf_ordering = True, batch_size = 1, verbose=1):
    
    image_shape = image.shape
    image = pad(image, patch_size)
    
    if verbose >= 1:
        print("New dimension {}".format(image.shape))
    
    segmentation = []
    for i in tqdm(range(0, image_shape[0]), disable = (verbose == 0)):
        if verbose >= 2:
            print("Slice: {}".format(i))
        
        patches = extract_patches_2d(image[i], patch_size)
        if not(tf_ordering):
            patches = np.transpose(patches, (0,3,2,1))

        #not sure if batch size matters
        predictions = np.empty((0,))
        i = 0
        while i < patches.shape[0]:
            if i + batch_size < patches.shape[0]:
                p = patches[i: i + batch_size]
            else:
                p = patches[i:]
            
            preds = classify(p, batch_size = p.shape[0], verbose=0)
            predictions = np.concatenate((predictions, preds))
            i += batch_size

        predictions = np.array(predictions)

        #transform linear array back into image
        slice = predictions.reshape(image_shape[1], image_shape[2])
        segmentation.append(slice)

    return np.array(segmentation)


if __name__ == '__main__':
    import keras
    from keras.models import load_model
    from loadImagesFromDisk import loadTestImage
    import sys

    assert(len(sys.argv) >= 4)
    model_path = sys.argv[1]
    image_dir_path = sys.argv[2]
    output_file = sys.argv[3]
    print("Output: ", output_file)

    tf_ordering = True
    if (keras.backend.image_dim_ordering() == "th"):
        tf_ordering = False
    print("Image ordering:", keras.backend.image_dim_ordering(), "tf_ordering", tf_ordering)

    model = load_model(model_path)

    image, image_dimension = loadTestImage(image_dir_path, use_N4Correction = True)

    normalize(image)

    batch_size = 128
    patch_size=(33,33)

    print("Image dimension", image_dimension)

    segmentation = segment(image, patch_size, model.predict_classes, tf_ordering = tf_ordering, batch_size = 128, verbose = 1)
    
    #save segmentation
    segmentation = sitk.GetImageFromArray(segmentation)
    segmentation = sitk.Cast(image, sitk.sitkUInt8)
    sitk.WriteImage(segmentation, output_file)


