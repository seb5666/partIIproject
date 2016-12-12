from loadImagesFromDisk import loadImages
from dataExtractor import DataExtractor
from visualise import showSlice
import numpy as np

(slices, labels, imageDimension) = loadImages('../Dataset/BRATS-2/Synthetic_Data/HG/')

print(slices.shape)
print(labels.shape)
print(imageDimension)

dataExtractor = DataExtractor(slices, labels, imageDimension)
X, y = dataExtractor.extractTrainingData(numSamples = 100)

print(X.shape)
print(y.shape)
