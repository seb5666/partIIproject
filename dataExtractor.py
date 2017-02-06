import numpy as np
from sklearn.utils import shuffle

from keras.utils import np_utils

class DataExtractor():
    
    def __init__(self, images, labels, validation_images, validation_labels, find_all_samples=False, tf_ordering=True):
      
        #Set to true if the number of samples to extract from the images should be the maximum possible, i.e the number of patches available for the least represented class
        self.find_all_samples = find_all_samples
        self.tf_ordering = tf_ordering

        print("Normalizing slices")
        for image in images:
            print("Normalizing training image", image.shape)
            for slice in image:
                slice = self.normalize(slice)
        for image in validation_images:
            print("Normalizing validation image", image.shape)
            for slice in image:
                slice = self.normalize(slice)
        print("Done normalizing")
        
        self.images = images
        self.labels = labels

        self.validation_images = validation_images
        self.validation_labels = validation_labels

        self.dimensions = np.array([image.shape for image in images])
        self.validation_dimensions = np.array([image.shape for image in validation_images])

    def extractTrainingData(self, training_samples = 10000, validation_samples = 1000, classes=[0,1,2,3,4]):
        patch_size = (33,33)

        X_train = []
        y_train = []
        X_val = []
        y_val = []

        if self.find_all_samples:
            samples_per_class = self.findValidSamplesNumber(self.images, self.labels, self.dimensions, patch_size, classes)
            validation_samples_per_class = self.findValidSamplesNumber(self.validation_images, self.validation_labels, self.validation_dimensions, patch_size, classes)
            print("Using", samples_per_class, "training samples per class")
            print("Using", validation_samples_per_class, "validation samples per class")
        else:
            samples_per_class = int(training_samples / len(classes))
            validation_samples_per_class = int(validation_samples / len(classes))
        

        for class_number in classes:
            train_p = []
            train_l = []
            if class_number == 0:
                train_p, train_l = self.findPatches(self.images, self.labels, self.dimensions, patch_size, 3 * samples_per_class, class_number)
            else:
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
	
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)

        return X_train, y_train, X_val, y_val

    def normalize_slice(self, slice):
        for mode in range(4):
            slice[:,:,mode] = normalize(slice[:,:,mode])
        return slice

    def normalize(self, slice):
        if np.std(slice) == 0:
            return slice
        else:
            return (slice - np.mean(slice)) / np.std(slice)

    def findValidSamplesNumber(self, images, labels, dimensions, patchSize, classes):
        min_so_far = None
        for class_number in classes:
            patch_coordinates = self.findValidPatchesCoordinates(images, labels, dimensions, patchSize, class_number)
            if min_so_far == None or min_so_far > len(patch_coordinates):
                min_so_far = len(patch_coordinates)
        return min_so_far
    
    def findValidPatchesCoordinates(self, images, labels, dimensions, patchSize, classNumber):
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

        return valid_centers

    def findPatches(self, images, labels, dimensions, patchSize, numPatches, classNumber):
        valid_centers = self.findValidPatchesCoordinates(images, labels, dimensions, patchSize, classNumber) 
        
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
            if not(self.tf_ordering):
                patch = np.transpose(patch)
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

