import numpy as np
import sys
import SimpleITK as sitk
from sklearn.metrics import classification_report

print(sys.argv)

if len(sys.argv) != 3:
    print("Please provide the segmented image and the ground truth images")

segmentation = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1]))
ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[2]))

print("Segmentation: ", segmentation.shape)
print("Ground truth: ", ground_truth.shape)

segmentation = segmentation.reshape((-1,))
ground_truth = ground_truth.reshape((-1,))
print("Segmentation: ", segmentation.shape)
print("Ground truth: ", ground_truth.shape)

print(classification_report(ground_truth, segmentation))

