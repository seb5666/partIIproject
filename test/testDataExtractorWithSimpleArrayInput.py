import unittest
from model.dataExtractor import DataExtractor
from test.testUtils import *

from collections import Counter
import numpy as np
from keras.utils import np_utils

class TestDataExtractorWithSimpleArrayInput(unittest.TestCase):

    def setUp(self):
        #self.images = [np.arange(4).reshape(1,2,2,1)]
        image1 =  np.array([[ [[0], [1]],
                              [[2], [3]] ]])
        self.images = [image1]

        self.labels = [np.array([[[0,1],[2,3]]])] 

        self.dimensions = np.array([image1.shape])

    def test_find_valid_patches_coordinates(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False)

        valid_training_patches = [dataExtractor.find_valid_patches_coordinates(self.images, self.labels, self.dimensions, class_number, patch_size) for class_number in [0,1,2,3,4]]
        #the patches should contain a patch for classes 0,1,2,3 but none for class 4
        self.assertTrue(contains(valid_training_patches[0], np.array([0, 0, 0, 0])))
        self.assertTrue(contains(valid_training_patches[1], np.array([0, 0, 0, 1]))) 
        self.assertTrue(contains(valid_training_patches[2], np.array([0, 0, 1, 0])))
        self.assertTrue(contains(valid_training_patches[3], np.array([0, 0, 1, 1]))) 
        self.assertTrue(valid_training_patches[0].shape[0]== 1) 
        self.assertTrue(valid_training_patches[1].shape[0]== 1) 
        self.assertTrue(valid_training_patches[2].shape[0]== 1) 
        self.assertTrue(valid_training_patches[3].shape[0]== 1) 
        self.assertTrue(valid_training_patches[4].shape[0]== 0) 
    
    def test_find_valid_patches_coordinates_with_large_patches(self):
        patch_size = (2,2)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False)

        valid_training_patches = [dataExtractor.find_valid_patches_coordinates(self.images, self.labels, self.dimensions, class_number, patch_size) for class_number in [0,1,2,3,4]]
        #valid_training_patches should only contain a single patch for class 3, the patch containing the entire image
        self.assertTrue(contains(valid_training_patches[3], np.array([0, 0, 1, 1]))) 
        self.assertTrue(valid_training_patches[3].shape[0] == 1) 
        self.assertTrue(valid_training_patches[0].shape[0] == 0) 
        self.assertTrue(valid_training_patches[1].shape[0] == 0) 
        self.assertTrue(valid_training_patches[2].shape[0] == 0) 
        self.assertTrue(valid_training_patches[4].shape[0] == 0) 
    
    def test_find_training_patches_close_to_tumour(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False)

        valid_training_patches = dataExtractor.find_patches_close_to_tumour(self.images, self.labels)
        #in this case all patches are within the bounding box of the tumour, so this should return all 4 patches, one for class 0,1,2 and 3
        self.assertTrue(contains(valid_training_patches[0], np.array([0, 0, 0, 0])))
        self.assertTrue(contains(valid_training_patches[1], np.array([0, 0, 0, 1]))) 
        self.assertTrue(contains(valid_training_patches[2], np.array([0, 0, 1, 0])))
        self.assertTrue(contains(valid_training_patches[3], np.array([0, 0, 1, 1]))) 
        self.assertTrue(valid_training_patches[4].size == 0) 

    def test_extract_training_patches(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False)
        
        train_X, train_y, val_X, val_y = dataExtractor.extractTrainingData(5)

        self.assertTrue(len(train_X) == 4)
        self.assertTrue(len(train_y) == 4)

        patches = np.array([[[[0]]], [[[1]]], [[[2]]], [[[3]]]])
        labels = np.concatenate([np_utils.to_categorical(0, 5), np_utils.to_categorical(1, 5), np_utils.to_categorical(2, 5), np_utils.to_categorical(3, 5)])

        for patch, label in zip(patches, labels):
            self.assertTrue(contains(train_X, patch)) 
            self.assertTrue(contains(train_y, label))
            self.assertTrue(index_of(train_X, patch) == index_of(train_y, label))

if __name__ == '__main__':
        unittest.main()
