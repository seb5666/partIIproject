import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import sys 
import os
from os import listdir

#Load all the training images and the labels that are within a high level directory (/HG/ for ex.)
def loadImages(images_directory):
    images = []
    labels = []
    dimensions = []
    for data_dir in sorted(listdir(images_directory)):
        if not(data_dir.startswith('.')):
            (image, label, dimension) = loadImage(os.path.join(images_directory, data_dir))
            images.append(image)
            labels.append(label)
            dimensions.append(dimension)
    return (images, labels, dimensions)

def loadImage(image_directory, use_N4Correction = True):
    ot = ""
    t1 = ""
    t1c = ""
    t2 = ""
    flair = ""
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "OT" in file:
                    ot = filePath
                elif "T1c" in file:
                    if use_N4Correction and "_normalized" in file:
                        t1c = filePath
                    if not(use_N4Correction) and "_normalized" not in file:
                        t1c = filePath
                elif "T1" in file:
                    if use_N4Correction and "_normalized" in file:
                        t1 = filePath
                    if not(use_N4Correction) and "_normalized" not in file:
                        t1 = filePath
                elif "T2" in file:
                    t2 = filePath
                elif "Flair" in file:
                    flair = filePath

    imagePaths = [t1, t1c, t2, flair] 

    images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(imagePath)) for imagePath in imagePaths]).astype('float32')
    
    if len(images) != 4 or ot == "":
        raise Exception("An error occured while reading from", image_directory)

    print("Reading images", imagePaths)
    print("Reading ground truth", ot)
    print("Read in an image of size", images.shape, "from", image_directory)

    OT = sitk.ReadImage(ot)
    labels = sitk.GetArrayFromImage(OT)

    #stack images along the 4th dimension so that image[z][y][x] containes the 4 values for the different scan types
    data = np.stack(images, axis=3)
    dimension = data.shape
    return (data, labels, dimension)

#Loads the 4 images but not the Observer Truth
def loadTestImage(image_directory):
    imagePaths = []
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "OT" not in file:
                    imagePaths.append(filePath)

    images = np.array([sitk.GetArrayFromImage(sitk.ReadImage(imagePath)) for imagePath in imagePaths]).astype('float32')

    if len(images) != 4:
        raise Exception("An error occured while reading from", image_directory)
    print("Reading images", imagePaths)

    #stack images along the 4th dimension so that image[z][y][x] containes the 4 values for the different scan types
    data = np.stack(images, axis=3)
    dimension = data.shape
    return (data, dimension)

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


