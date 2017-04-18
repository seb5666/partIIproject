import sys
import os
from os import listdir

import numpy as np
import scipy.stats

import SimpleITK as sitk 
from nipype.interfaces.ants import N4BiasFieldCorrection 
from subprocess import call

def winsorize(image_path, output_path):
    print("Winsorizing {}. Saved to file {}".format(image_path, output_path))
    scan = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    scan = scipy.stats.mstats.winsorize(scan, limits=0.01)
    sitk.WriteImage(sitk.GetImageFromArray(scan), output_path)

def applyN4(image_path, output_path):
    print("Applying N4 correction {}. Saved to file {}".format(image_path, output_path))
    n4 = N4BiasFieldCorrection()
    n4.inputs.n_iterations = [20, 20, 20, 10]
    n4.inputs.dimension = 3
    n4.inputs.bspline_fitting_distance = 200
    n4.inputs.shrink_factor = 2
    n4.inputs.convergence_threshold = 0
    n4.inputs.input_image = image_path
    n4.inputs.output_image = output_path
    os.system(n4.cmdline)


data_dir = sys.argv[1]
output_dir = sys.argv[2]

for patient_dir in sorted(listdir(data_dir)):
    if patient_dir.startswith("."):
        continue
    t1 = ""
    t1c = ""
    t2 = ""
    flair = ""
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(data_dir,patient_dir)):
        for file in filenames:
            if file.endswith('.mha'):
                filePath = os.path.join(dirpath, file)
                if "T1c" in file:
                    if "_normalized" not in file:
                        t1c = filePath
                elif "T1" in file:
                    if "_normalized" not in file:
                        t1 = filePath
                elif "T2" in file:
                    t2 = filePath
                elif "Flair" in file:
                        flair = filePath

    output_path = os.path.join(output_dir, patient_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output1_path = output_path + "/T1_winsorised.mha"
    output2_path = output_path + "/T1_corrected.mha"
    winsorize(t1, output1_path)
    #applyN4(t1, output2_path)

    output1_path = output_path + "/T1c_winsorised.mha"
    output2_path = output_path + "/T1c_corrected.mha"
    winsorize(t1c, output1_path)
    #applyN4(t1c, output2_path)

    output1_path = output_path + "/T2_winsorised.mha"
    output2_path = output_path + "/T2_corrected.mha"
    winsorize(t2, output1_path)
    #applyN4(t2, output2_path)

    output1_path = output_path + "/Flair_winsorised.mha"
    output2_path = output_path + "/Flair_corrected.mha"
    winsorize(flair, output1_path)
    #applyN4(flair, output2_path)
