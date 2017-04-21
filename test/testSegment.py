import unittest
import numpy as np
from model.segment import segment
from model.segment import pad


class TestSegment(unittest.TestCase):

    def test_pad(self):
        image = np.full(shape=(1,2,2,4), fill_value = 1)
    
        patch_size = (1,1)
        self.assertTrue(pad(image, patch_size).shape == image.shape)

        patch_size = (33,33)
        shape = pad(image,patch_size).shape
        self.assertTrue(image.shape[0] == shape[0])
        self.assertTrue(image.shape[1] + 32 == shape[1])
        self.assertTrue(image.shape[2] + 32 == shape[2])
        self.assertTrue(image.shape[3] == shape[3])
    
        patch_size = (2,2)
        padded = pad(image, patch_size)
        image2 = np.full(padded.shape, 0)
        image2[0, 1:3, 1:3 , :] = 1
        self.assertTrue(np.array_equal(padded, image2))

    def test_segment_to_constant(self):
        patch_size = (1,1)
        classify_zero = lambda data, batch_size, verbose : [0 for i in range(batch_size)]

        image = np.random.random(size=(1,1,1,4))
        segmentation = segment(image, patch_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        

        image = np.random.random(size=(4,4,4,4))
        segmentation = segment(image, patch_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        
        segmentation = segment(image, patch_size, classify_zero, batch_size = 4, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 
        
        patch_size = (3,3)
        image = np.random.random(size=(4,4,4,4))
        segmentation = segment(image, patch_size, classify_zero, verbose = 0)
        self.assertTrue(np.array_equal(segmentation, np.full(segmentation.shape, 0)))
        self.assertTrue(segmentation.shape == image.shape[0:3]) 

if __name__ == '__main__':
    unittest.main()
