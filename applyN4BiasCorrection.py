import sys
import os
from os import listdir
import numpy as np
import SimpleITK as sitk

data_dir = sys.argv[1]

def correctImages(images_directory):
    for data_dir in sorted(listdir(images_directory)):
        if not(data_dir.startswith('.')):
                correctImage(os.path.join(images_directory, data_dir))

def correctImage(image_directory):
    print(image_directory)
    t1 = ""
    t1c = ""
    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "T1c" in file:
                        t1c = filePath
                elif "T1" in file:
                        t1 = filePath

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    T1_image = sitk.Cast(sitk.ReadImage(t1), sitk.sitkFloat32)
    T1c_image = sitk.Cast(sitk.ReadImage(t1c), sitk.sitkFloat32)

    T1_mask = sitk.OtsuThreshold(T1_image, 0, 1, 200)
    T1c_mask = sitk.OtsuThreshold(T1c_image, 0, 1, 200)

    T1_n4 = corrector.Execute(T1_image, T1_mask)
    print("Corrected", t1)

    T1c_n4 = corrector.Execute(T1c_image, T1c_mask)
    print("Corrected", t1c)

    sitk.WriteImage(T1_n4, t1[:-4] + '_normalized.mha')
    sitk.WriteImage(T1c_n4, t1c[:-4] + '_normalized.mha')

correctImages(data_dir)
