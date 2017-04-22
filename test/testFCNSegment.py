import unittest
import numpy as np
from test.testUtils import contains
from model.fcn_segment import pad, segment, extract_patches

class TestFCNSegment(unittest.TestCase):
   
    def setUp(self):
        self.image1 = np.array([[
                    [[0],[0],[1],[1]],
                    [[0],[0],[1],[1]],
                    [[2],[2],[3],[3]],
                    [[2],[2],[3],[3]]]])

    def test_extract_patches(self):
        patch_size = (2,2)
        label_size = (2,2)
        patches = [np.full((2,2,1), np.array([i])) for i in range(4)]
        extracted_patches = extract_patches(self.image1[0], self.image1.shape, patch_size, label_size)
        for patch in patches:
            self.assertTrue(contains(extracted_patches, patch))
        
        patch_size = (4,4)
        label_size = (2,2)
        image2 = pad(self.image1, patch_size, label_size)
        extracted_patches = extract_patches(image2[0], self.image1.shape, patch_size, label_size)
        centers = extracted_patches[:,1:3,1:3,:]
        for patch in patches:
            self.assertTrue(contains(centers, patch))
    
    def test_pad(self):
    
        patch_size = (1,1)
        label_size = (1,1)
        padded = pad(self.image1, patch_size, label_size)
        self.assertTrue(self.image1.shape[0] == padded.shape[0])
        self.assertTrue(self.image1.shape[1] + patch_size[0] == padded.shape[1])
        self.assertTrue(self.image1.shape[2] + patch_size[1] == padded.shape[2])
        self.assertTrue(self.image1.shape[3] == padded.shape[3])

        patch_size = (10,10)
        label_size = (1,1)
        padded = pad(self.image1, patch_size, label_size)
        half_width = int(patch_size[0]/2) - label_size[0]
        half_height = int(patch_size[1]/2) - label_size[1]
        center = padded[:, half_width : half_width + self.image1.shape[1], half_height : half_height + self.image1.shape[2], :]
        self.assertTrue(np.array_equal(center, self.image1))

    def test_segment_to_constant(self):
        image = np.random.random((1,4,4,4))
        patch_size = (2,2)
        label_size = (2,2)
        classify_zero = lambda data, batch_size, verbose : [np.full(label_size[0] * label_size[1], 0) for i in range(batch_size)]
        segmentation = segment(image, image.shape, patch_size, label_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        
        patch_size = (4,4)
        label_size = (2,2)
        classify_zero = lambda data, batch_size, verbose : [np.full(label_size[0] * label_size[1], 0) for i in range(batch_size)]
        segmentation = segment(image, image.shape, patch_size, label_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        
        image = np.random.random((1,32,32,4)) 

        patch_size = (64,64)
        label_size = (8,8)
        classify_zero = lambda data, batch_size, verbose : [np.full(label_size[0] * label_size[1], 0) for i in range(batch_size)]
        segmentation = segment(image, image.shape, patch_size, label_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        
        patch_size = (5,5)
        label_size = (3,3)
        classify_zero = lambda data, batch_size, verbose : [np.full(label_size[0] * label_size[1], 0) for i in range(batch_size)]
        segmentation = segment(image, image.shape, patch_size, label_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 

if __name__ == '__main__':
    unittest.main()
