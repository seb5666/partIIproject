import unittest
from model.dataExtractor import DataExtractor
from test.testUtils import *

from collections import Counter
import numpy as np
from keras.utils import np_utils

#scan with 2 modalities, none of the voxels of patch 0 are within the bounding box of the tumourous region
class TestDataExtractorWithLargerArrayInput(unittest.TestCase):

    def setUp(self):
        image1 =  np.array([[ [[0,0], [0,0], [0,0]],
                              [[1,1], [2,2], [0,0]],
                              [[3,3], [4,4], [0,0]]]])
        self.images = [image1]

        self.labels = [np.array([[[0,0,0],
                                  [1,2,0],
                                  [3,4,0]]] )]

        self.dimensions = np.array([image1.shape])

    def test_find_valid_patches_coordinates(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False, normalization = None)

        valid_training_patches = [dataExtractor.find_valid_patches_coordinates(self.images, self.labels, self.dimensions, class_number, patch_size) for class_number in [0,1,2,3,4]]
        #the patches should contain 5 patches for class 0 and 1 patch for each of 1,2,3,4
        self.assertTrue(valid_training_patches[0].shape[0]== 5) 
        self.assertTrue(valid_training_patches[1].shape[0]== 1) 
        self.assertTrue(valid_training_patches[2].shape[0]== 1) 
        self.assertTrue(valid_training_patches[3].shape[0]== 1) 
        self.assertTrue(valid_training_patches[4].shape[0]== 1) 
    
    def test_find_valid_patches_coordinates_with_large_patches(self):
        patch_size = (2,2)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False, normalization = None)

        valid_training_patches = [dataExtractor.find_valid_patches_coordinates(self.images, self.labels, self.dimensions, class_number, patch_size) for class_number in [0,1,2,3,4]]
        #valid_training_patches should only contain a single patch for class 2 and 4, and 2 patches for class 0 
        self.assertTrue(valid_training_patches[0].shape[0] == 2) 
        self.assertTrue(valid_training_patches[1].shape[0] == 0) 
        self.assertTrue(valid_training_patches[2].shape[0] == 1) 
        self.assertTrue(valid_training_patches[3].shape[0] == 0) 
        self.assertTrue(valid_training_patches[4].shape[0] == 1) 
    
    def test_find_training_patches_close_to_tumour(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False, normalization = None)

        valid_training_patches = dataExtractor.find_patches_close_to_tumour(self.images, self.labels)
        #the only voxels inside the region close to the tumour are those for classes 1,2,3,4
        self.assertTrue(valid_training_patches[0].shape[0]== 0) 
        self.assertTrue(valid_training_patches[1].shape[0]== 1) 
        self.assertTrue(valid_training_patches[2].shape[0]== 1) 
        self.assertTrue(valid_training_patches[3].shape[0]== 1) 
        self.assertTrue(valid_training_patches[4].shape[0]== 1) 

    def test_extract_training_patches(self):
        patch_size = (1,1)
        label_size = (1,1)
        dataExtractor = DataExtractor(self.images, self.labels, patch_size = patch_size, label_size = label_size, distance_between_patches_in_class0 = False, validation_samples_per_class = 0, num_channels = 1, verbose = False, normalization = None)
        
        train_X, train_y, val_X, val_y = dataExtractor.extractTrainingData(5)

        self.assertTrue(len(train_X) == 5)
        self.assertTrue(len(train_y) == 5)

        patches = np.array([[[[0, 0]]], [[[1, 1]]], [[[2, 2]]], [[[3, 3]]]])
        labels = np.concatenate([np_utils.to_categorical(0, 5), np_utils.to_categorical(1, 5), np_utils.to_categorical(2, 5), np_utils.to_categorical(3, 5)])

        for patch, label in zip(patches, labels):
            self.assertTrue(contains(train_X, patch)) 
            self.assertTrue(contains(train_y, label))
            self.assertTrue(index_of(train_X, patch) == index_of(train_y, label))

if __name__ == '__main__':
        unittest.main()
