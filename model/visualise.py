import matplotlib.pyplot as plt
from loadImagesFromDisk import loadTestImage
import sys

def showSlice(slices, labels, sliceNumber):
        plt.figure(1)
        plt.imshow(labels[sliceNumber,:,:])
        plt.legend("Observer truth")
        plt.figure(2)
        plt.imshow(slices[sliceNumber,:,:,0])
        plt.figure(3)
        plt.imshow(slices[sliceNumber,:,:,1])
        plt.figure(4)
        plt.imshow(slices[sliceNumber,:,:,2])
        plt.figure(5)
        plt.imshow(slices[sliceNumber,:,:,3])
        plt.show()

image_dir_path = sys.argv[1]
slices, dimensions = loadTestImage(image_dir_path, use_N4Correction = False)
plt.imshow(slices[77,:,:,1])
plt.show()
