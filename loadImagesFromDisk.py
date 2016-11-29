import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import sys 
import os
from os import listdir

def loadImages(image_directory):
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
        raise Exception("An error occured while the data was read in!")

    OT = sitk.ReadImage(observerTruthPath)
    labels = sitk.GetArrayFromImage(OT)
    #stack images along the 4th dimension so that image[z][y][x] containes the 4 values for the different scan types
    data = np.stack(images, axis=3)

    return (data, labels)

def showImage(image, sliceNumber=0):
    imgArray = sitk.GetArrayFromImage(image)
    plt.imshow(imgArray[sliceNumber,: ,:])
    plt.show()

# Load all images
dirPath = sys.argv[1]
(data, labels) = loadImages(dirPath)

print(labels.shape)
print(labels[155,155,:])
print(data.shape)
print(data[155,155,:])
#showImage(OT, 155)


