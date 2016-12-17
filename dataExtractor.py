import numpy as np
from keras.utils import np_utils
class DataExtractor():
    
    def __init__(self, slices, labels, dimension):
        self.slices = slices
        self.labels = labels
        self.imageZ = dimension[0]
        self.imageY = dimension[1]
        self.imageX = dimension[2]

    def findPatches(self, patchSize, numPatches, classNumber):
        possible_centers = np.argwhere(self.labels == classNumber)

        #remove positions whose patches don't fit in the image
        possible_centers = self.filterValidPositions(possible_centers, patchSize)

        #randomly choose numPatches valid center_pixels
        indexes = np.random.choice(possible_centers.shape[0], numPatches, replace=False)
        centers = possible_centers[indexes, :]

        #extract patches around those center pixels
        return self.createPatches(centers, patchSize, classNumber)

    def createPatches(self, centers, patchSize, classNumber):
        patches = []
        
        for center_pixel in centers:
            startRow = center_pixel[1] - int(patchSize[0]/2)
            endRow = center_pixel[1] + int(patchSize[0]/2) + 1
            startCol = center_pixel[2] - int(patchSize[1]/2)
            endCol = center_pixel[2] + int(patchSize[1]/2) + 1
            patches.append(self.slices[center_pixel[0], startRow:endRow, startCol:endCol, :])
        
        l = np.full(len(patches), classNumber, dtype='float')
        return np.array(patches), l

    def filterValidPositions(self, possible_centers, patchSize):
        halfHeight = int(patchSize[0]/2)
        halfWidth = int(patchSize[1]/2)
        #merge all four conditions into one for performance improvement
        possible_centers = possible_centers[possible_centers[:,1] - halfHeight >= 0]
        possible_centers = possible_centers[possible_centers[:,1] + halfHeight + 1 < self.imageY]
        possible_centers = possible_centers[possible_centers[:,2] - halfWidth >= 0]
        possible_centers = possible_centers[possible_centers[:,2] + halfWidth + 1 < self.imageX]
        return possible_centers

    def extractTrainingData(self, numSamples = 100000, classes=[0,1,2,3,4]):
        X = []
        y = []
        samplesPerClass = int(numSamples / len(classes))
        for classNumber in classes:
            p, l = self.findPatches((33,33), samplesPerClass, classNumber)
            X.append(p)
            y.append(l)
        y = np.concatenate(y)
        y = np_utils.to_categorical(y, int(len(classes)))
        return np.concatenate(X), y
