import unittest
import numpy as np
from model.normalization import normalize_channel, normalize_scans

class TestNormalization(unittest.TestCase):

    def test_normalize_channel(self):
        channel = np.random.random(size=(2,2,2))
        normalized_channel = normalize_channel(channel)
        self.assertTrue(channel.shape == normalized_channel.shape)
        self.assertTrue(np.isclose(np.std(normalized_channel), 1))
        self.assertTrue(np.isclose(np.mean(normalized_channel), 0))

        channel = np.random.random(size=(150,150,150))
        normalized_channel = normalize_channel(channel)
        self.assertTrue(channel.shape == normalized_channel.shape)
        self.assertTrue(np.isclose(np.std(normalized_channel), 1))
        self.assertTrue(np.isclose(np.mean(normalized_channel), 0))
       
        #stddev is 0 so don't change the array
        channel = np.full(shape=(2,2,2), fill_value = 0)
        normalized_channel = normalize_channel(channel)
        self.assertTrue(np.array_equal(channel, normalized_channel))
    
    def test_normalize_scans(self):
        num_channels = 4
        images = [np.random.random(size=(15,15,15, num_channels)) for i in range(10)]
        normalized_images = normalize_scans(images, num_channels)
        for image, normalized_image in zip(images, normalized_images):
            self.assertTrue(normalized_image.shape == image.shape)
            self.assertTrue(np.isclose(np.std(normalized_image), 1))
            self.assertTrue(np.isclose(np.mean(normalized_image), 0))
if __name__ == '__main__':
    unittest.main()
