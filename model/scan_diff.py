import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys

scan1 = sys.argv[1]
scan2 = sys.argv[2]

image1 = sitk.GetArrayFromImage(sitk.ReadImage(scan1))
image2 = sitk.GetArrayFromImage(sitk.ReadImage(scan2))

sliceNumber = 90
slice1 = image1[sliceNumber,:,:]
slice2 = image2[sliceNumber,:,:]
slice3 = slice1 - slice2

combined = np.array([slice1, slice2, slice3])

_min, _max = np.amin(combined), np.amax(combined)

cmap = 'nipy_spectral'
plt.subplot(131)
plt.imshow(slice1, cmap=cmap, vmin=_min, vmax=_max)
plt.subplot(132)
plt.imshow(slice2, cmap=cmap, vmin=_min, vmax=_max)
plt.subplot(133)
plt.imshow(slice3, cmap=cmap, vmin=_min, vmax=_max)
plt.colorbar()
plt.show()

plt.imshow(slice3)
plt.show()
