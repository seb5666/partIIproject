import unittest
from model.dataExtractor import DataExtractor

import numpy as np

class TestPatchExtraction(unittest.TestCase):

    def setUp(self):
        images = []
        labels = []
        val_images = []
        val_labes = []
        patch_size = (2,2)
        label_size = (1,1)
        self.dataExtractor = DataExtractor(images, labels, val_images, val_labels, patch_size = patch_size, label_size = label_size)
        return
        print("Setting up")

    def tearDown(self):
        return
        print("Tear down")

    def test_a(self):
        self.assertEqual(1, 0 + 1)

    def test_b(self):
        self.assertEqual(1,1)

    

if __name__ == '__main__':
        unittest.main()
