import unittest
import numpy as np
from model.remove_connected_components import remove_connected_components


class TestRemoveConnectedComponents(unittest.TestCase):

    def test_no_components(self):
        image = np.full(shape=(3,3,3), fill_value=0)
        image2 = np.copy(image)
        remove_connected_components(image2, 1, verbose=False)
        self.assertTrue(np.array_equal(image, image2))
    
    def test_remove_one_component(self):
        image = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                          [[0,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]])
        remove_connected_components(image, 2, verbose=False)
        self.assertTrue(np.array_equal(image, np.full(image.shape, 0)))

    def test_remove_large_3d_component(self):
        image = np.array([[[3,3,0],
                           [0,4,0],
                           [0,0,0]],

                          [[0,0,0],
                           [0,2,0],
                           [0,0,0]],

                          [[0,0,0],
                           [1,1,0],
                           [2,2,0]]])
        remove_connected_components(image, image.size, verbose=False)
        self.assertTrue(np.array_equal(image, np.full(image.shape, 0)))
    
    def test_diagonals(self):
        image = np.array([[[1,1,0],
                           [0,0,1],
                           [0,0,0]]])

        image2 = np.array([[[1,1,0],
                           [0,0,0],
                           [0,0,0]]])
        remove_connected_components(image, 2, verbose=False)
        self.assertTrue(np.array_equal(image, image2))

    def test_threshold(self):
        image = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                          [[0,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]])
        image2 = np.copy(image)
        remove_connected_components(image2, 0, verbose=False)
        self.assertTrue(np.array_equal(image, image2))
    
    def test_remove_multiple_components(self):
        image = np.array([[[3,3,0],
                           [0,4,0],
                           [0,0,0]],

                          [[0,0,0],
                           [0,0,0],
                           [0,0,0]],

                          [[0,0,0],
                           [1,1,0],
                           [2,2,0]]])
        remove_connected_components(image, 5, verbose=False)
        self.assertTrue(np.array_equal(image, np.full(image.shape, 0)))
if __name__ == '__main__':
    unittest.main()
