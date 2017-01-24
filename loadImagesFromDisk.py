import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import sys 
import os
from os import listdir

#Load all the training images and the labels that are within a high level directory (/HG/ for ex.)
def loadImages(images_directory, use_N4Correction = True):
    images = []
    labels = []
    dimensions = []
    for data_dir in sorted(listdir(images_directory)):
        if not(data_dir.startswith('.')):
            OT = loadGroundTruth(os.path.join(images_directory, data_dir))
            scans = loadScans(os.path.join(images_directory, data_dir), use_N4Correction)
            image = np.array([sitk.GetArrayFromImage(scan) for scan in scans]).astype('float32')
            image, dimension = stackImage(image)
            label = sitk.GetArrayFromImage(OT)
            
            print("Read in an image of size", image.shape, "from", images_directory)

            images.append(image)
            labels.append(label)
            dimensions.append(dimension)

    return (images, labels, dimensions)

def loadScans(image_directory, use_N4Correction = True):
    t1 = ""
    t1c = ""
    t2 = ""
    flair = ""
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "T1c" in file:
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
    images = [sitk.ReadImage(imagePath) for imagePath in imagePaths]

    if len(images) != 4:
        raise Exception("An error occured while reading from", image_directory)

    print("Reading images", imagePaths)

    return images

def loadGroundTruth(image_directory):
    ot = ""
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "OT" in file:
                    ot = filePath

    if ot == "":
        raise Exception("No ground truth image was found in", image_directory)

    OT = sitk.ReadImage(ot)
    print("Reading ground truth", ot)
    return OT

    #stack images along the 4th dimension so that image[z][y][x] containes the 4 values for the different scan types
def stackImage(images):
    data = np.stack(images, axis=3)
    dimension = data.shape
    return (data, dimension)

#Loads the 4 images but not the Observer Truth
def loadTestImage(image_directory, use_N4Correction = True):
    scans = loadScans(image_directory, use_N4Correction)
    image = np.array([sitk.GetArrayFromImage(scan) for scan in scans]).astype('float32')
    images, dimension = stackImage(image)
    return (images, dimension)

def showImage(image, sliceNumber=0):
    plt.imshow(image[sliceNumber,: ,:])
    plt.show()

# Load all images
# dirPath = sys.argv[1]


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


