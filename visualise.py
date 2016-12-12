import matplotlib.pyplot as plt

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
