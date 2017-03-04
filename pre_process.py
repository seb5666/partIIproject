import sys
import os
from os import listdir

import numpy as np
import scipy.stats

import SimpleITK as sitk

from loadImagesFromDisk import loadScans, stackImage

images_dir = sys.argv[1]
output_dir = sys.argv[2]

N4_corrector = sitk.N4BiasFieldCorrectionImageFilter()

images = []
for data_dir in sorted(listdir(images_dir)):
    if not(data_dir.startswith('.')):
        scans = loadScans(os.path.join(images_dir, data_dir), use_N4Correction = False)
        image = np.array([sitk.GetArrayFromImage(scan) for scan in scans]).astype('float32')
        
        print("Read in an image of size", image.shape, "from", images_dir)
        for scan in image:
            print("max before winsorizing", np.max(scan))
            scan = scipy.stats.mstats.winsorize(scan, limits=0.01)
            print("max after winsorizing", np.max(scan))

        t1 = image[0]
        t1c = image[1]
        t2 = image[2]
        flair = image[3]
        
        t1 = sitk.GetImageFromArray(t1)
        t1c = sitk.GetImageFromArray(t1c)
        t2 = sitk.GetImageFromArray(t2)
        flair = sitk.GetImageFromArray(flair)
        
        save_dir = os.path.join(output_dir, data_dir)
        if not os.path.exists(save_dir):
            print("Creating directory", save_dir)
            os.makedirs(save_dir)

        #Apply N4 normalization
        print("Applying N4 to T1 and T1c images in ", data_dir)
        
        t1_mask = sitk.OtsuThreshold(t1, 0, 1, 200)
        t1 = N4_corrector.Execute(t1, t1_mask)
        print("Corrected t1")
        sitk.WriteImage(t1, save_dir + "/T1_normalized.mha")

        t1c_mask = sitk.OtsuThreshold(t1c, 0, 1, 200)
        t1c = N4_corrector.Execute(t1c, t1c_mask)
        print("Corrected t1c")
        sitk.WriteImage(t1c, save_dir + "/T1c_normalized.mha")

        #t2_mask = sitk.OtsuThreshold(t2, 0, 1, 200)
        #t2 = N4_corrector.Execute(t2, t2_mask)
        #print("Corrected t2")
        sitk.WriteImage(t2, save_dir + "/T2_normalized.mha")
        
        #flair_mask = sitk.OtsuThreshold(flair, 0, 1, 200)
        #flair = N4_corrector.Execute(flair, flair_mask)
        #print("Corrected flair")
        sitk.WriteImage(flair, save_dir + "/Flair_normalized.mha")
        

