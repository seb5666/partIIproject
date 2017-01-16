import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from loadImagesFromDisk import loadTestImage
import sys
import os
from os import listdir

assert(len(sys.argv) >= 3)

dirPath = sys.argv[1]
patchSize=(33,33)
model = load_model(sys.argv[2])
print("Model shape", model.input_shape)

(data, dimension) = loadTestImage(dirPath)

#normalize data
for slice in data:
    if np.max(slice) != 0:
        slice /= np.max(slice)

segmentation = np.zeros(data.shape[0:3])

#extract patches + padd for edges?
halfPatchHeight= int(patchSize[0]/2)
halfPatchWidth = int(patchSize[1]/2)
print(dimension)
imageHeight = dimension[1]
imageWidth= dimension[2]

#for z in range(0, dimension[0]):
for z in range(72, 85):
    print("Slice", z)
    for y in range(halfPatchHeight, imageHeight - halfPatchHeight):
        print("y", y)
        for x in range(halfPatchWidth, imageWidth - halfPatchWidth):
            patch = data[z,y-halfPatchHeight : y + halfPatchHeight + 1, x - halfPatchWidth: x + halfPatchWidth + 1, :]
            prediction = model.predict_classes(np.array([patch]), batch_size=1, verbose=0)[0]
            segmentation[z, y, x] = prediction
            print(prediction, end=', ')

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

showSegmentation(segmentation)

