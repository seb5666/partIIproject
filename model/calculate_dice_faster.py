import numpy as np
import sys
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix

print(sys.argv)

if len(sys.argv) != 3:
    print("Please provide the segmented image and the ground truth images")

def dice_score(confusion):
    return 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])

def evaluate_region(segmentation, ground_truth, positive, negative):
    print("Evaluating for positive {}, negative {}".format(positive, negative))
    s = np.copy(segmentation)
    gt = np.copy(ground_truth)

    for x in negative:
        np.place(s, s == x, 0)
        np.place(gt, gt == x, 0)
    for x in positive:
        np.place(s, s == x, 1)
        np.place(gt, gt == x, 1)

    C = confusion_matrix(gt, s)
    print(C)
    print(dice_score(C))

segmentation = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1]))
ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[2]))

print("Segmentation: ", segmentation.shape)
print("Ground truth: ", ground_truth.shape)

segmentation = segmentation.reshape((-1,))
ground_truth = ground_truth.reshape((-1,))

print(confusion_matrix(ground_truth, segmentation))

evaluate_region(segmentation, ground_truth, [1,2,3,4], [0])
evaluate_region(segmentation, ground_truth, [1,3,4], [0,2])
evaluate_region(segmentation, ground_truth, [4], [0,1,2,3])
