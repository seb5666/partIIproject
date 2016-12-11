import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import sys 
import os
from os import listdir

def loadImages(images_directory):
    images = []
    labels = []
    for data_dir in listdir(images_directory):
        if not(data_dir.startswith('.')):
            (image, label) = loadImage(os.path.join(images_directory, data_dir))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return (images, labels)

def loadImage(image_directory):
    imagePaths = []
    observerTruthPath = ""
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "OT" in file:
                    observerTruthPath = filePath
                else:
                    imagePaths.append(filePath)
    images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(imagePath)) for imagePath in imagePaths])
    
    if len(images) != 4 or observerTruthPath == "":
        raise Exception("An error occured while reading from", image_directory)
    print("Read in an image of size", images.shape, "from", image_directory)

    OT = sitk.ReadImage(observerTruthPath)
    labels = sitk.GetArrayFromImage(OT)

    #stack images along the 4th dimension so that image[z][y][x] containes the 4 values for the different scan types
    data = np.stack(images, axis=3)

    return (data, labels)

def showImage(image, sliceNumber=0):
    plt.imshow(image[sliceNumber,: ,:])
    plt.show()

# Load all images
# dirPath = sys.argv[1]
# (data, labels) = loadImages(dirPath)

#print("Labels", labels.shape)
#print("Data", data.shape)
#slice = 150
#plt.figure(1)
#plt.imshow(labels[2,slice,:,:])
#plt.figure(2)
#plt.imshow(data[2,slice,:,:,0])
#plt.figure(3)
#plt.imshow(data[2,slice,:,:,1])
#plt.figure(4)
#plt.imshow(data[2,slice,:,:,2])
#plt.figure(5)
#plt.imshow(data[2,slice,:,:,3])
#plt.show()


