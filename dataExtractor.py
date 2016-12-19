import numpy as np
from sklearn.utils import shuffle

from keras.utils import np_utils

class DataExtractor():
    
    def __init__(self, images, labels, dimensions):
        self.images = images
        self.labels = labels
        self.dimensions = np.array(dimensions)

        # this should potentially be done somewhere else. Also not sure if it should be applied at a slice level or image level..
        print("Normalizing slices")
        for image in images:
            print("Normalizing image", image.shape)
            for slice in image:
                if np.max(slice) != 0:
                    slice /= np.max(slice)
        print("Done normalizing")

    def findPatches(self, patchSize, numPatches, classNumber):
        possible_centers = []

        #create list of voxel indexes [image, z, y, x] matching the classNumber
        for index, image in enumerate(self.labels):
            centers = np.argwhere(image == classNumber)
            indexes = np.full((centers.shape[0], 1), index, dtype="int")
            possible_centers.append(np.append(indexes, centers, axis = 1))
        possible_centers = np.concatenate(possible_centers)
        print("Possible for", classNumber, possible_centers.shape)
        
        possible_centers = self.filterValidPositions(possible_centers, patchSize)
        print("Valid for", classNumber, possible_centers.shape)
        
        #randomly choose numPatches valid center_pixels
        indexes = np.random.choice(possible_centers.shape[0], numPatches, replace=False)
        centers = possible_centers[indexes, :]

        #extract patches around those center pixels
        p, l = self.createPatches(centers, patchSize, classNumber)
        print("Patches", p.shape)
        # this returns copies of p and l which is not ideal, create a method to do it in place?
        return shuffle(p, l)

    def createPatches(self, centers, patchSize, classNumber):
        patches = []
        
        for center_pixel in centers:
            startRow = center_pixel[2] - int(patchSize[0]/2)
            endRow = center_pixel[2] + int(patchSize[0]/2) + 1
            startCol = center_pixel[3] - int(patchSize[1]/2)
            endCol = center_pixel[3] + int(patchSize[1]/2) + 1
            image = self.images[center_pixel[0]]
            patch = image[center_pixel[1], startRow:endRow, startCol:endCol, :]
            patches.append(patch)
        
        l = np.full(len(patches), classNumber, dtype='int')
        return np.array(patches), l

    def filterValidPositions(self, possible_centers, patchSize):
        halfHeight = int(patchSize[0]/2)
        halfWidth = int(patchSize[1]/2)

        #merge all four conditions into one for performance improvement
        possible_centers = possible_centers[possible_centers[:,2] - halfHeight >= 0]
        possible_centers = possible_centers[possible_centers[:,2] + halfHeight + 1 < self.dimensions[possible_centers[:,0]][:,1]]
        possible_centers = possible_centers[possible_centers[:,3] - halfWidth >= 0]
        possible_centers = possible_centers[possible_centers[:,3] + halfWidth + 1 < self.dimensions[possible_centers[:,0]][:,2]]
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
