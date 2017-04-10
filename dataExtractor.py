import numpy as np
from sklearn.utils import shuffle

from keras.utils import np_utils

class DataExtractor():
    
    def __init__(self, images, labels, validation_images, validation_labels, find_all_samples=False, tf_ordering=True, normalization = "scan", classes=[0,1,2,3,4], patch_size = (33,33), label_size = (1,1)):
      
        #Set to true if the number of samples to extract from the images should be the maximum possible, i.e the number of patches available for the least represented class
        self.find_all_samples = find_all_samples

        self.tf_ordering = tf_ordering

        self.normalization = normalization
        
        self.classes = classes
        
        self.patch_size = patch_size
        self.label_size = label_size

        self.images = images
        self.labels = labels

        self.validation_images = validation_images
        self.validation_labels = validation_labels
        
        self.dimensions = np.array([image.shape for image in images])
        self.validation_dimensions = np.array([image.shape for image in validation_images])
        
        print("Finding valid training patches")
        self.valid_training_patches = [self.findValidPatchesCoordinates(self.images, self.labels, self.dimensions, classNumber, self.patch_size) for classNumber in self.classes]
        self.valid_training_patches_close_to_tumours = self.find_patches_close_to_tumour(self.images, self.labels)

#        for classNumber in self.classes:
#            print("Class " , classNumber)
#            print(self.valid_training_patches[classNumber].shape)
#            print(self.valid_training_patches_close_to_tumours[classNumber].shape)
#            for i in range(4):
#                print("Dimension = ", i)
#                bincount = np.bincount(self.valid_training_patches[classNumber][:,i])
#                bincount2 = np.bincount(self.valid_training_patches_close_to_tumours[classNumber][:,i])
#                print(bincount.shape)
#                print(bincount)
#                print(bincount2.shape)
#                print(bincount2)

        print("Finding valid validation patches")
#        self.valid_validation_patches = [self.findValidPatchesCoordinates(self.validation_images, self.validation_labels, self.validation_dimensions, classNumber, self.patch_size) for classNumber in self.classes]
        self.valid_validation_patches = self.find_patches_close_to_tumour(self.validation_images, self.validation_labels)


        if self.normalization == "scan":
            num_channels = 4
            print("Transfroming each scan to [0,1] range and normalizing each scan")
            #print(self.images[0][70,:,:,0])
            for image in self.images:
                for channel in range(num_channels):
                    sequence = image[:,:,:, channel]
                    image[:,:,:, channel] = (sequence - np.min(sequence)) /(np.max(sequence) - np.min(sequence))
                    image[:,:,:, channel] = self.normalize_channel(image[:,:,:, channel])
            for image in self.validation_images:
                for channel in range(4):
                    sequence = image[:,:,:, channel]
                    image[:,:,:, channel] = (sequence - np.min(sequence)) /(np.max(sequence) - np.min(sequence))
                    image[:,:,:, channel] = self.normalize_channel(image[:,:,:, channel])
            #print(self.images[0][70,:,:,0])
            print("Done normalizing")

        if self.normalization == "dataset":
            print("Normalizing with dataset mean and stds")
            num_channels = 4
            epsilon = 0.0001
           
            channel_values = np.concatenate([self.images[i][:,:,:,0].reshape((-1)) for i in range(len(images))])
            print(channel_values.shape)
            means = [np.mean(np.concatenate([self.images[i][:,:,:,c].reshape((-1)) for i in range(len(images))]).reshape((-1))) for c in range(num_channels)]
            stds = [np.std(np.concatenate([self.images[i][:,:,:,c].reshape((-1)) for i in range(len(images))]).reshape((-1))) for c in range(num_channels)]
            print("Channel means on training data", means)
            print("Channel std on training data", stds)
            
            print("Normalising each channel")
            #print(self.images[0][70,70,70,0])
            for image in self.images:
                for i in range(num_channels):
                    channel = image[:,:,:,i]
                    image[:,:,:, i] = (channel - means[i]) / (stds[i] + epsilon)
            for image in self.validation_images:
                for i in range(4):
                    channel = image[:,:,:,i]
                    image[:,:,:, i] = (channel - means[i]) / (stds[i] + epsilon)
            print("Done normalizing")
            #print(self.images[0][70,70,70,0])
    
    def find_patches_close_to_tumour(self, images, labels):
        #Some analysis save to remove
        tumour_mins = []
        tumour_maxs = []
        for label in labels:
            mins = [np.min(np.argwhere(label > 0)[:, i]) for i in range(3)]
            maxs = [np.max(np.argwhere(label > 0)[:, i]) + 1 for i in range(3)]
            mins[1] = max(int(self.patch_size[0]/2), mins[1])
            mins[2] = max(int(self.patch_size[1]/2), mins[2])
            maxs[1] = min(label.shape[1] - int(self.patch_size[0]/2) - 1, maxs[1])
            maxs[2] = min(label.shape[2] - int(self.patch_size[1]/2) - 1, maxs[2])
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
            print("Valid for", classNumber, valid_centers.shape)
            
            valid_training_patches.append(valid_centers)
        return valid_training_patches
            
    def extractRandomTrainingData(self, training_samples = 9000, validation_samples=1000):
        training_centers = []
        for classNumber in self.classes:
            training_centers.append(self.valid_training_patches[classNumber])
        training_centers = np.concatenate(training_centers)
        print("Possible training centers", training_centers.shape)

        training_indexes = np.random.choice(training_centers.shape[0], training_samples, replace=False)
        training_centers = training_centers[training_indexes, :]

        print("Training centers shape", training_centers.shape)

        #extract patches around those center pixels
        X_train = self.extractPatches(self.images, training_centers, self.patch_size)
        y_train, count = self.extractLabels(self.labels, training_centers)
        print("Training data distribution {}".format(count))
       
        #extract equiprobable validation data
        validation_samples_per_class = int(validation_samples / len(self.classes))
        X_val = []
        y_val = []
        validation_class_distribution = []
        for class_number in self.classes:
            valid_centers = self.valid_validation_patches[class_number]
        
            #randomly choose numPatches valid center_pixels
            indexes = np.random.choice(valid_centers.shape[0], validation_samples_per_class, replace=False)
            centers = valid_centers[indexes, :]
            
            #extract patches around those center pixels
            p = self.extractPatches(self.validation_images, centers, self.patch_size)
            l, count = self.extractLabels(self.validation_labels, centers)
            print(l.shape)
            
            X_val.append(p)
            y_val.append(l)
            validation_class_distribution.append(count)
        
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        
        print("Training data shape{}, training labels shape {}".format(X_train.shape, y_train.shape))
        print("Validation data shape{}, validation labels shape {}".format(X_val.shape, y_val.shape))
        
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)

        return X_train, y_train, X_val, y_val

    def extractTrainingData(self, training_samples = 10000, validation_samples = 1000, classes=[0,1,2,3,4], samples_weights = [1,1,1,1,1]):

        print("Extracting %d training_samples and %d validation_samples, weights"%(training_samples, validation_samples), samples_weights)

        X_train = []
        y_train = []
        X_val = []
        y_val = []

        if self.find_all_samples:
            samples_per_class = self.findValidSamplesNumber(self.valid_training_patches, self.dimensions, self.patch_size, classes)
            validation_samples_per_class = self.findValidSamplesNumber(self.valid_validation_patches, self.validation_dimensions, self.patch_size, classes)
            print("Using", samples_per_class, "training samples per class")
            print("Using", validation_samples_per_class, "validation samples per class")
        else:
            samples_per_class = int(training_samples / len(classes))
            validation_samples_per_class = int(validation_samples / len(classes))
        
        for class_number in classes:
            train_p = []
            train_l = []
            
            if class_number == 0:
                #choose a fraction close to tumours only
                train_p, train_l = self.findPatches(self.valid_training_patches_close_to_tumours, self.images, self.dimensions, samples_per_class * samples_weights[class_number], class_number)
                train_p2, train_l2 = self.findPatches(self.valid_training_patches, self.images, self.dimensions, samples_per_class * samples_weights[class_number], class_number)
                
                p = 0.5 # fraction from close to tumour patches
                n = int(samples_per_class * samples_weights[class_number] * p)
                
                indexes1 = np.random.choice(np.arange(len(train_p)), n, replace = False)
                indexes2 = np.random.choice(np.arange(len(train_p2)), samples_per_class * samples_weights[class_number] - n, replace = False)
                
                train_p, train_l = train_p[indexes1], train_l[indexes1]
                train_p2, train_l2 = train_p2[indexes2], train_l2[indexes2]
                
                train_p = np.concatenate((train_p, train_p2))
                train_l = np.concatenate((train_l, train_l2))
            
            else:
                train_p, train_l = self.findPatches(self.valid_training_patches_close_to_tumours, self.images, self.dimensions, samples_per_class * samples_weights[class_number], class_number)

            X_train.append(train_p)
            y_train.append(train_l)

            val_p, val_l = self.findPatches(self.valid_validation_patches, self.validation_images, self.validation_dimensions, validation_samples_per_class, class_number)
            X_val.append(val_p)
            y_val.append(val_l)

        y_train = np.concatenate(y_train)
        y_train = np_utils.to_categorical(y_train, int(len(classes)))
        X_train = np.concatenate(X_train)

        y_val = np.concatenate(y_val)
        y_val = np_utils.to_categorical(y_val, int(len(classes)))
        X_val = np.concatenate(X_val)
        
        if self.normalization == "all_training_patches":
            print("Normalizing with training patches means and stds")
            num_channels = 4
            epsilon = 0.0001
           
            means = [np.mean(np.concatenate([X_train[:,:,:,c].reshape((-1))])) for c in range(num_channels)]
            stds = [np.std(np.concatenate([X_train[:,:,:,c].reshape((-1))])) for c in range(num_channels)]
            print("Training patches means on training data", means)
            print("Training patches std on training data", stds)
            
            print("Normalising each patch")
            print(X_train[0,:,:,0])
            for patch in X_train:
                for c in range(num_channels):
                    patch[:,:,c] = (patch[:,:,c] - means[c]) / stds[c]
            for patch in X_val:
                for c in range(num_channels):
                    patch[:,:,c] = (patch[:,:,c] - means[c]) / stds[c]
            print("Done normalizing")
            print(X_train[0,:,:,0])

        if self.normalization == "individual_patches":
            print("Normalizing with individual patches means and stds")
            print("Normalising patches")
            X_train = self.normalize_patches(X_train)
            X_val = self.normalize_patches(X_val)
            print("Done normalising patches")

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
        
            #randomly choose numPatches valid center_pixels
            indexes = np.random.choice(valid_centers.shape[0], validation_samples_per_class, replace=False)
            centers = valid_centers[indexes, :]
            
            #extract patches around those center pixels
            p = self.extractPatches(self.validation_images, centers, self.patch_size)
            l, count = self.extractLabels(self.validation_labels, centers)
            
            X_val.append(p)
            y_val.append(l)
            validation_class_distribution.append(count)
        
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        
        validation_class_distribution = np.sum(validation_class_distribution, axis = 0)

        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)

        print("Training class distribution {}, {}".format(training_class_distribution, ["{0:0.3f}".format(t / sum(training_class_distribution)) for t in training_class_distribution]))
        print("Validation class distribution {}, {}".format(validation_class_distribution, ["{0:0.3f}".format(t / sum(validation_class_distribution)) for t in validation_class_distribution]))

        print("Training data shape{}, training labels shape {}".format(X_train.shape, y_train.shape))
        print("Validation data shape{}, validation labels shape {}".format(X_val.shape, y_val.shape))

        return X_train, y_train, X_val, y_val

    def extractDoubleTrainingData(self, training_samples = 10000, validation_samples = 1000, classes=[0,1,2,3,4], small_patch_size=(16,16)):
        
        print("Extracting %d training_samples and %d validation_samples"%(training_samples, validation_samples))
            
        samples_per_class = int(training_samples / len(classes))
        X_train1 = []
        X_train2 = []
        y_train = []
        training_class_distribution = []
        for class_number in classes:
            valid_centers = shuffle(self.valid_training_patches[class_number])
            
            #randomly choose numPatches valid center_pixels
            indexes = np.random.choice(valid_centers.shape[0], samples_per_class, replace=False)
            centers = valid_centers[indexes, :]
            
            #extract patches around those center pixels
            p1 = self.extractPatches(self.images, centers, self.patch_size)
            p2 = self.extractPatches(self.images, centers, small_patch_size)
            l, count = self.extractLabels(self.labels, centers)
            
            X_train1.append(p1)
            X_train2.append(p2)
            y_train.append(l)
            training_class_distribution.append(count)

        X_train1 = np.concatenate(X_train1)
        X_train2 = np.concatenate(X_train2)
        X_train = [X_train2, X_train1]
        y_train = np.concatenate(y_train)
        
        training_class_distribution = np.sum(training_class_distribution, axis = 0)
        
        validation_samples_per_class = int(validation_samples / len(classes))
        X_val1 = []
        X_val2 = []
        y_val = []
        validation_class_distribution = []
        for class_number in classes:
            valid_centers = self.valid_validation_patches[class_number]
            
            #randomly choose numPatches valid center_pixels
            indexes = np.random.choice(valid_centers.shape[0], validation_samples_per_class, replace=False)
            centers = valid_centers[indexes, :]
            
            #extract patches around those center pixels
            p1 = self.extractPatches(self.validation_images, centers, self.patch_size)
            p2 = self.extractPatches(self.validation_images, centers, small_patch_size)
            l, count = self.extractLabels(self.validation_labels, centers)
            
            X_val1.append(p1)
            X_val2.append(p2)
            y_val.append(l)
            validation_class_distribution.append(count)
        
        X_val1 = np.concatenate(X_val1)
        X_val2 = np.concatenate(X_val2)
        X_val = [X_val2, X_val1]
        y_val = np.concatenate(y_val)
        
        validation_class_distribution = np.sum(validation_class_distribution, axis = 0)
        
        print("Training class distribution {}, {}".format(training_class_distribution, ["{0:0.3f}".format(t / sum(training_class_distribution)) for t in training_class_distribution]))
        print("Validation class distribution {}, {}".format(validation_class_distribution, ["{0:0.3f}".format(t / sum(validation_class_distribution)) for t in validation_class_distribution]))
        
        print("Training data shape{}, training labels shape {}".format([X_train[0].shape, X_train[1].shape], y_train.shape))
        print("Validation data shape{}, validation labels shape {}".format([X_val[0].shape, X_val[1].shape], y_val.shape))
        
        return X_train, y_train, X_val, y_val


    def normalize_patches(self, patches):
        if self.tf_ordering:
            for p_index, patch in enumerate(patches):
                for channel in range(patch.shape[-1]):
                    patches[p_index, :, :, channel] = self.normalize_channel(patch[:, :, channel])
        else:
            for p_index, patch in enumerate(patches):
                for channel in range(patch.shape[0]):
                    patches[p_index, channel, :, :] = self.normalize_channel(patch[channel, :, :])
        return patches
    
    def normalize_channel(self, channel):
        std = np.std(channel)
        if std != 0:
            return (channel - np.mean(channel)) / std
        else:
            return channel

    def findValidSamplesNumber(self, valid_patches, dimensions, classes):
        min_so_far = None
        for class_number in classes:
            patch_coordinates = valid_patches[class_number]
            if min_so_far == None or min_so_far > len(patch_coordinates):
                min_so_far = len(patch_coordinates)
        return min_so_far
    
    def findValidPatchesCoordinates(self, images, labels, dimensions, classNumber, patch_size):
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
        
        valid_centers = self.filterValidPositions(dimensions, possible_centers, patch_size)
        print("Valid for", classNumber, valid_centers.shape)

        #if classNumber == 0:
        #    valid_centers = valid_centers[valid_centers[:,1] % 3 == 0]
        #    valid_centers = valid_centers[valid_centers[:,2] % 3 == 0]
        #    valid_centers = valid_centers[valid_centers[:,3] % 3 == 0]
        #    print("Valid for after removing_neighbours", classNumber, valid_centers.shape)
                
        return valid_centers

    def findPatches(self, valid_patches, images, dimensions, numPatches, classNumber):
        valid_centers = valid_patches[classNumber]
        
        #randomly choose numPatches valid center_pixels
        indexes = np.random.choice(valid_centers.shape[0], numPatches, replace=False)
        centers = valid_centers[indexes, :]
        
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
        #merge all four conditions into one for performance improvement
        possible_centers = possible_centers[possible_centers[:,2] - halfHeight >= 0]
        possible_centers = possible_centers[possible_centers[:,2] + halfHeight + 1 < dimensions[possible_centers[:,0]][:,1]]
        possible_centers = possible_centers[possible_centers[:,3] - halfWidth >= 0]
        possible_centers = possible_centers[possible_centers[:,3] + halfWidth + 1 < dimensions[possible_centers[:,0]][:,2]]
        
        return possible_centers

