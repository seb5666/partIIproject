from loadImagesFromDisk import loadImages
import numpy as np
import matplotlib.pyplot as plt

(images, labels) = loadImages('../Dataset/BRATS-2/Synthetic_Data/HG/')

print(images.shape)
print(labels.shape)

def showSlice(imageNumber, slice):
    plt.figure(1)
    plt.imshow(labels[imageNumber,slice,:,:])
    plt.legend("Observer truth")
    plt.figure(2)
    plt.imshow(images[imageNumber,slice,:,:,0])
    plt.figure(3)
    plt.imshow(images[imageNumber,slice,:,:,1])
    plt.figure(4)
    plt.imshow(images[imageNumber,slice,:,:,2])
    plt.figure(5)
    plt.imshow(images[imageNumber,slice,:,:,3])
    plt.show()

def createPatches(imageNumber, layerNumber, patchSize):
    image = images[imageNumber, layerNumber, :, :, :]
    patches = [] 
    patch_labels = []
    for startingRow in range(image.shape[0] - patchSize[0] + 1):
        for startingColumn in range(image.shape[1] - patchSize[1] + 1):
            endRow = startingRow + patchSize[0]
            endColumn = startingColumn + patchSize[1]
            patches.append(image[startingRow:endRow,startingColumn:endColumn,:])
            patch_labels.append(labels[imageNumber, layerNumber, startingRow + int(patchSize[0]/2), startingColumn + int(patchSize[1]/2)])
    patches = np.array(patches)
    patch_labels = np.array(patch_labels)
    print(patches.shape)
    print(patch_labels.shape)
    return (patches, patch_labels)


patches = []
imageNumber = 2
for layerNumber in range(0,len(images[imageNumber]), 15):
    patches.append(createPatches(imageNumber, layerNumber, (33,33)))
    print("Done layer", layerNumber)
print(type(patches))
print(patches.shape)
patches = np.array(patches)
print(patches.size)
