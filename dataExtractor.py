import numpy as np
from sklearn.utils import shuffle

from keras.utils import np_utils

class DataExtractor():
    
    def __init__(self, images, labels, dimensions, validation_images=2):
        
        # this should potentially be done somewhere else. Also not sure if it should be applied at a slice level or image level..
        print("Normalizing slices")
        for image in images:
            print("Normalizing image", image.shape)
            for slice in image:
                if np.max(slice) != 0:
                    slice /= np.max(slice)
        print("Done normalizing")
        
        self.images = images[:-validation_images]
        self.labels = labels[:-validation_images]

        self.validation_images = images[-validation_images:]
        self.validation_labels = labels[-validation_images:]

        dimensions = np.array(dimensions)
        self.dimensions = dimensions[:-validation_images]
        self.validation_dimensions = dimensions[-validation_images:]

    
    def findPatches(self, images, labels, dimensions, patchSize, numPatches, classNumber):
        possible_centers = []

        #create list of voxel indexes [image, z, y, x] matching the classNumber
        for index, image in enumerate(labels):
            centers = np.argwhere(image == classNumber)
            #create image indexes
            indexes = np.full((centers.shape[0], 1), index, dtype="int")
            #append image indexes with center indexes
            possible_centers.append(np.append(indexes, centers, axis = 1))

        possible_centers = np.concatenate(possible_centers)
        print("Possible for", classNumber, possible_centers.shape)
        
        valid_centers = self.filterValidPositions(dimensions, possible_centers, patchSize)
        print("Valid for", classNumber, valid_centers.shape)
        
        #randomly choose numPatches valid center_pixels
        indexes = np.random.choice(valid_centers.shape[0], numPatches, replace=False)
        centers = valid_centers[indexes, :]

        #extract patches around those center pixels
        p, l = self.createPatches(images, centers, patchSize, classNumber)
        print("Patches", p.shape)
        # this returns copies of p and l which is not ideal, create a method to do it in place?
        return shuffle(p, l)

    def createPatches(self, images, centers, patchSize, classNumber):
        patches = []
        
        for center_pixel in centers:
            startRow = center_pixel[2] - int(patchSize[0]/2)
            endRow = center_pixel[2] + int(patchSize[0]/2) + 1
            startCol = center_pixel[3] - int(patchSize[1]/2)
            endCol = center_pixel[3] + int(patchSize[1]/2) + 1
            image = images[center_pixel[0]]
            patch = image[center_pixel[1], startRow:endRow, startCol:endCol, :]
            patches.append(patch)
        
        l = np.full(len(patches), classNumber, dtype='int')
        return np.array(patches), l

    def filterValidPositions(self, dimensions, possible_centers, patchSize):
        halfHeight = int(patchSize[0]/2)
        halfWidth = int(patchSize[1]/2)
        #merge all four conditions into one for performance improvement
        possible_centers = possible_centers[possible_centers[:,2] - halfHeight >= 0]
        possible_centers = possible_centers[possible_centers[:,2] + halfHeight + 1 < dimensions[possible_centers[:,0]][:,1]]
        possible_centers = possible_centers[possible_centers[:,3] - halfWidth >= 0]
        possible_centers = possible_centers[possible_centers[:,3] + halfWidth + 1 < dimensions[possible_centers[:,0]][:,2]]
        return possible_centers

    def extractTrainingData(self, training_samples = 100000, validation_samples = 1000, classes=[0,1,2,3,4]):
        X_train = []
        y_train = []
        samples_per_class = int(training_samples / len(classes))

        X_val = []
        y_val = []
        validation_samples_per_class = int(validation_samples / len(classes))
        
        patch_size = (33,33)

        for class_number in classes:
            train_p, train_l = self.findPatches(self.images, self.labels, self.dimensions, patch_size, samples_per_class, class_number)
            X_train.append(train_p)
            y_train.append(train_l)

            val_p, val_l = self.findPatches(self.validation_images, self.validation_labels, self.validation_dimensions, patch_size, validation_samples_per_class, class_number)
            X_val.append(val_p)
            y_val.append(val_l)

        y_train = np.concatenate(y_train)
        y_train = np_utils.to_categorical(y_train, int(len(classes)))
        X_train = np.concatenate(X_train)

        y_val = np.concatenate(y_val)
        y_val = np_utils.to_categorical(y_val, int(len(classes)))
        X_val = np.concatenate(X_val)

        print("training patches shape", X_train.shape)
        print("training labels shape", y_train.shape)

        print("val patches shape", X_val.shape)
        print("val labels shape", y_val.shape)

        return X_train, y_train, X_val, y_val
