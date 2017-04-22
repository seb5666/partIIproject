import numpy as np
from sklearn.utils import shuffle

from keras.utils import np_utils

from normalization import normalize_scans

class DataExtractor():
    
    def __init__(self, images, labels, validation_samples_per_class = 100, tf_ordering=True, normalization = "scan", classes=[0,1,2,3,4], patch_size = (33,33), label_size = (1,1), distance_between_patches_in_class0 = True, num_channels = 4, verbose = True):
      
        self.tf_ordering = tf_ordering

        self.normalization = normalization
        
        self.classes = classes
        
        self.distance_between_patches_in_class0 = distance_between_patches_in_class0
        self.patch_size = patch_size
        self.label_size = label_size
        self.num_channels = num_channels
        self.verbose = verbose
        
        self.validation_samples_per_class = validation_samples_per_class
        self.images = images
        self.labels = labels
        
        self.dimensions = np.array([image.shape for image in images])
        
        self.debug("Finding valid training patches")
        self.valid_training_patches = [self.find_valid_patches_coordinates(self.images, self.labels, self.dimensions, classNumber, self.patch_size) for classNumber in self.classes]
        self.valid_training_patches_close_to_tumours = self.find_patches_close_to_tumour(self.images, self.labels)

        if validation_samples_per_class > 0:

            self.debug("Finding {} per class valid validation patches".format(validation_samples_per_class))
            indexes = [(np.random.choice(len(self.valid_training_patches_close_to_tumours[classNumber]), validation_samples_per_class, replace = False) if len(self.valid_training_patches_close_to_tumours[classNumber]) > 0 else np.array([])) for classNumber in self.classes]
            
            self.valid_validation_patches = np.array([self.valid_training_patches_close_to_tumours[classNumber][indexes[classNumber]] for classNumber in self.classes])
            
            #Remove validation patches from training sets
            self.debug(list(map(lambda l : l.shape, self.valid_training_patches_close_to_tumours)))
            self.debug(list(map(lambda l : l.shape, self.valid_training_patches)))
            self.valid_training_patches_close_to_tumours = np.array([np.delete(self.valid_training_patches_close_to_tumours[classNumber], indexes[classNumber], axis = 0) for classNumber in self.classes])
            to_remove = []
            found = 0
            for j in range(len(self.valid_validation_patches[0])):
                val_patch = self.valid_validation_patches[0][j]
                zs = np.argwhere(self.valid_training_patches[0][:,0] == val_patch[0])[:,0]
                ys = np.argwhere(self.valid_training_patches[0][:,1] == val_patch[1])[:,0]
                xs = np.argwhere(self.valid_training_patches[0][:,2] == val_patch[2])[:,0]
                ws = np.argwhere(self.valid_training_patches[0][:,3] == val_patch[3])[:,0]
                
                inter = np.intersect1d(zs, np.intersect1d(ys, np.intersect1d(xs, ws)))
                if len(inter)>0:
                    found += 1
                    print(inter)
                    to_remove.extend(inter)
            self.debug("Removing {} overlapping patches".format(len(to_remove)))
            self.valid_training_patches = np.array([np.delete(self.valid_training_patches[0], to_remove, axis = 0)])
            
            self.debug(list(map(lambda l : l.shape, self.valid_training_patches_close_to_tumours)))
            self.debug(list(map(lambda l : l.shape, self.valid_training_patches)))

            self.debug("Normalizing each scan")
            self.images = normalize_scans(self.images, self.num_channels)

    def find_patches_close_to_tumour(self, images, labels):
        tumour_mins = []
        tumour_maxs = []
        for label in labels:
            mins = [np.min(np.argwhere(label > 0)[:, i]) for i in range(3)]
            maxs = [np.max(np.argwhere(label > 0)[:, i]) + 1 for i in range(3)]
            mins[1] = max(int(self.patch_size[0]/2), mins[1])
            mins[2] = max(int(self.patch_size[1]/2), mins[2])
            maxs[1] = min(label.shape[1] - int(self.patch_size[0]/2), maxs[1])
            maxs[2] = min(label.shape[2] - int(self.patch_size[1]/2), maxs[2])
            tumour_mins.append(mins)
            tumour_maxs.append(maxs)
        
        cut_images = []
        cut_labels = []
        for image, label, mins, maxs in zip(images, labels, tumour_mins, tumour_maxs):
            cut_image = image[mins[0]: maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            cut_label = label[mins[0]: maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            cut_images.append(cut_image)
            cut_labels.append(cut_label)
        cut_dimensions = np.array([image.shape for image in cut_images])

        valid_training_patches = []
        
        for classNumber in self.classes:
            possible_centers = []
            #create list of voxel indexes [image, z, y, x] matching the classNumber
            for index, image in enumerate(cut_labels):
                centers = np.argwhere(image == classNumber)
                #add offset to match the actual position of the entire image
                centers += tumour_mins[index]
                
                #create image indexes
                indexes_column = np.full((centers.shape[0], 1), index, dtype="int")
                #append image indexes with center indexes
                possible_centers.append(np.append(indexes_column, centers, axis = 1))
            
            valid_centers = np.concatenate(possible_centers)
            
            valid_training_patches.append(valid_centers)
        return valid_training_patches
    
    def extractTrainingData(self, training_samples = 10000):

        self.debug("Extracting {} training_samples".format(training_samples))

        X_train = []
        y_train = []
        X_val = []
        y_val = []

        samples_per_class = int(training_samples / len(self.classes))
        for class_number in self.classes:
            train_p = []
            train_l = []
            
            
            if len(self.valid_training_patches_close_to_tumours[class_number] > 0):
                if class_number == 0 and len(self.valid_training_patches_close_to_tumours[class_number]) >= 2:
                        train_p, train_l = self.findPatches(self.valid_training_patches_close_to_tumours, self.images, self.dimensions, int(samples_per_class/2), class_number)
                        train_p2, train_l2 = self.findPatches(self.valid_training_patches, self.images, self.dimensions, int(samples_per_class/2), class_number)
                        
                        train_p = np.concatenate((train_p, train_p2))
                        train_l = np.concatenate((train_l, train_l2))
                else:
                    train_p, train_l = self.findPatches(self.valid_training_patches_close_to_tumours, self.images, self.dimensions, samples_per_class, class_number)
                X_train.append(train_p)
                y_train.append(train_l)
            
            if self.validation_samples_per_class > 0:
                val_p, val_l = (self.createPatches(self.images, self.valid_validation_patches[class_number], class_number))
            else:
                val_p, val_l = np.array([]), np.array([])
            X_val.append(val_p)
            y_val.append(val_l)
                
        y_train = np.concatenate(y_train)
        y_train = np_utils.to_categorical(y_train, int(len(self.classes)))
        X_train = np.concatenate(X_train)

        y_val = np.concatenate(y_val)
        y_val = np_utils.to_categorical(y_val, int(len(self.classes)))
        X_val = np.concatenate(X_val)

        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)

        return X_train, y_train, X_val, y_val
    
    def extractLargeTrainingData(self, training_samples = 10000, validation_samples = 1000, classes=[0,1,2,3,4]):
        
        print("Extracting %d training_samples and %d validation_samples"%(training_samples, validation_samples))
        
        samples_per_class = int(training_samples / len(classes))
        X_train = []
        y_train = []
        training_class_distribution = []
        for class_number in classes:
            if class_number == 0:
                valid_centers = self.valid_training_patches[class_number]
                valid_centers_close_to_tumours= self.valid_training_patches_close_to_tumours[class_number]
                
                p = 0.5 # fraction from close to tumour patches
                n = int(samples_per_class * p)
                
                indexes = np.random.choice(valid_centers.shape[0], n, replace = False)
                centers = valid_centers[indexes, :]
                
                #makes sure we don't train on validation patches
                while True:
                    found = 0
                    for j in range(len(self.valid_validation_patches[class_number])):
                        val_patch = self.valid_validation_patches[class_number][j]
                        a = np.argwhere(centers[:,0] == val_patch[0])[:,0]
                        b = np.argwhere(centers[:,1] == val_patch[1])[:,0]
                        c = np.argwhere(centers[:,2] == val_patch[2])[:,0]
                        d = np.argwhere(centers[:,3] == val_patch[3])[:,0]
                        
                        inter = np.intersect1d(a, np.intersect1d(b, np.intersect1d(c, d)))
                        if len(inter)>0:
                            found += 1
                    if found == 0:
                        break
                    else:
                        print("Selected {} overlaping patches, need to reselect them for class {}".format(found, class_number))
                        indexes = np.random.choice(valid_centers.shape[0], n, replace = False)
                        centers = valid_centers[indexes, :]
            
                indexes_close_to_tumours = np.random.choice(valid_centers_close_to_tumours.shape[0], samples_per_class - n, replace = False)
                centers_close_to_tumours = valid_centers_close_to_tumours[indexes_close_to_tumours, :]

                centers = np.concatenate((centers, centers_close_to_tumours))
            else:
                valid_centers = self.valid_training_patches[class_number]
                #randomly choose numPatches valid center_pixels
                indexes = np.random.choice(valid_centers.shape[0], samples_per_class, replace=False)
            
                centers = valid_centers[indexes, :]

            #extract patches around those center pixels
            p = self.extractPatches(self.images, centers, self.patch_size)
            l, count = self.extractLabels(self.labels, centers)
            
            X_train.append(p)
            y_train.append(l)
            training_class_distribution.append(count)

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        training_class_distribution = np.sum(training_class_distribution, axis = 0)
        
        validation_samples_per_class = int(validation_samples / len(classes))
        X_val = []
        y_val = []
        validation_class_distribution = []
        for class_number in classes:
            valid_centers = self.valid_validation_patches[class_number]
            
            #extract patches around those center pixels
            p = self.extractPatches(self.images, valid_centers, self.patch_size)
            l, count = self.extractLabels(self.labels, valid_centers)
            
            X_val.append(p)
            y_val.append(l)
            validation_class_distribution.append(count)
        
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        
        validation_class_distribution = np.sum(validation_class_distribution, axis = 0)

        X_train, y_train = shuffle(X_train, y_train)

        print("Training class distribution {}, {}".format(training_class_distribution, ["{0:0.3f}".format(t / sum(training_class_distribution)) for t in training_class_distribution]))
        print("Validation class distribution {}, {}".format(validation_class_distribution, ["{0:0.3f}".format(t / sum(validation_class_distribution)) for t in validation_class_distribution]))

        print("Training data shape{}, training labels shape {}".format(X_train.shape, y_train.shape))
        print("Validation data shape{}, validation labels shape {}".format(X_val.shape, y_val.shape))

        return X_train, y_train, X_val, y_val
    
    def normalize_channel(self, channel):
        std = np.std(channel)
        if std != 0:
            return (channel - np.mean(channel)) / std
        else:
            return channel
    
    def find_valid_patches_coordinates(self, images, labels, dimensions, classNumber, patch_size):
        possible_centers = []
        #create list of voxel indexes [image, z, y, x] matching the classNumber
        for index, image in enumerate(labels):
            centers = np.argwhere(image == classNumber)
            #create image indexes
            indexes = np.full((centers.shape[0], 1), index, dtype="int")
            #append image indexes with center indexes
            possible_centers.append(np.append(indexes, centers, axis = 1))
        
        possible_centers = np.concatenate(possible_centers)
        self.debug("Possible for {} {}".format(classNumber, possible_centers.shape))
        valid_centers = self.filterValidPositions(dimensions, possible_centers, patch_size)
        self.debug("Valid for {} {}".format(classNumber, valid_centers.shape))

        if self.distance_between_patches_in_class0 and classNumber == 0:
            valid_centers = valid_centers[valid_centers[:,1] % 3 == 0]
            valid_centers = valid_centers[valid_centers[:,2] % 3 == 0]
            valid_centers = valid_centers[valid_centers[:,3] % 3 == 0]
            self.debug("Valid for after removing_neighbours {} {}".format(classNumber, valid_centers.shape))
                
        return valid_centers

    def findPatches(self, valid_patches, images, dimensions, numPatches, classNumber):
        valid_centers = valid_patches[classNumber]
        if valid_centers.shape[0] > 0:
            #randomly choose numPatches valid center_pixels
            indexes = np.random.choice(valid_centers.shape[0], numPatches, replace=False)
            centers = valid_centers[indexes, :]
            
            #makes sure we don't train on validation patches
            if self.validation_samples_per_class > 0:
                while True:
                    found = 0
                    for j in range(len(self.valid_validation_patches[classNumber])):
                        val_patch = self.valid_validation_patches[classNumber][j]
                        a = np.argwhere(centers[:,0] == val_patch[0])[:,0]
                        b = np.argwhere(centers[:,1] == val_patch[1])[:,0]
                        c = np.argwhere(centers[:,2] == val_patch[2])[:,0]
                        d = np.argwhere(centers[:,3] == val_patch[3])[:,0]
                        
                        inter = np.intersect1d(a, np.intersect1d(b, np.intersect1d(c, d)))
                        if len(inter)>0:
                            found += 1
                            print(inter)
                    if found == 0:
                        break
                    else:
                        print("Selected {} overlaping patches, need to reselect them for class {}".format(found, classNumber))
                        indexes = np.random.choice(valid_centers.shape[0], numPatches, replace=False)
                        centers = valid_centers[indexes, :]
        else:
            centers = valid_centers
        #extract patches around those center pixels
        p, l = self.createPatches(images, centers, classNumber)
        # this returns copies of p and l which is not ideal, create a method to do it in place?
        return shuffle(p, l)
    
    def extractPatches(self, images, centers, patch_size):
        patches = []
        for center_pixel in centers:
            startRow = center_pixel[2] - int(patch_size[0]/2)
            endRow = center_pixel[2] + int(patch_size[0]/2) + (patch_size[0] % 2)
            startCol = center_pixel[3] - int(patch_size[1]/2)
            endCol = center_pixel[3] + int(patch_size[1]/2) + (patch_size[0] % 2)
            image = images[center_pixel[0]]
            patch = image[center_pixel[1], startRow:endRow, startCol:endCol, :]
            if not(self.tf_ordering):
                patch = np.transpose(patch)
            patches.append(patch)
        
        return np.array(patches)

    def extractLabels(self, ground_truth_images, centers):
        labels = []
        count = [0 for c in self.classes]
        for center_pixel in centers:
            startRow = center_pixel[2] - int(self.label_size[0]/2)
            endRow = center_pixel[2] + int(self.label_size[0]/2) + (self.label_size[0] % 2)
            startCol = center_pixel[3] - int(self.label_size[1]/2)
            endCol = center_pixel[3] + int(self.label_size[1]/2) + (self.label_size[0] % 2)
            image = ground_truth_images[center_pixel[0]]
            patch = image[center_pixel[1], startRow:endRow, startCol:endCol]
            #Using to categorical flattens the 2d array in a 1d array
            count = [count[i] + np.count_nonzero(patch == self.classes[i]) for i in range(len(self.classes))]
            label = np_utils.to_categorical(patch, len(self.classes))
            labels.append(label)
        return np.array(labels), np.array(count)

    def createPatches(self, images, centers, classNumber):
        patches = []
        for center_pixel in centers:
            startRow = center_pixel[2] - int(self.patch_size[0]/2)
            endRow = center_pixel[2] + int(self.patch_size[0]/2) + 1
            startCol = center_pixel[3] - int(self.patch_size[1]/2)
            endCol = center_pixel[3] + int(self.patch_size[1]/2) + 1
            image = images[center_pixel[0]]
            patch = image[center_pixel[1], startRow:endRow, startCol:endCol, :]
            if not(self.tf_ordering):
                patch = np.transpose(patch)
            patches.append(patch)
        
        l = np.full(len(patches), classNumber, dtype='int')
        return np.array(patches), l

    def filterValidPositions(self, dimensions, possible_centers, patch_size):
        if patch_size == (0,0):
            return possible_centers
        
        halfHeight = int(patch_size[0]/2)
        halfWidth = int(patch_size[1]/2)

        possible_centers = possible_centers[possible_centers[:,2] >= halfHeight]
        possible_centers = possible_centers[possible_centers[:,2] <= dimensions[possible_centers[:,0]][:,1] - (halfHeight + (patch_size[0] % 2))]
        possible_centers = possible_centers[possible_centers[:,3] >= halfWidth]
        possible_centers = possible_centers[possible_centers[:,3] <= dimensions[possible_centers[:,0]][:,2] - (halfWidth + (patch_size[0] % 2))]

        return possible_centers

    def debug(self, str):
        if self.verbose:
            print(str)
