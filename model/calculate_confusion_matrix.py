import numpy as np
import sys
import SimpleITK as sitk

import os

from sklearn.metrics import confusion_matrix

def dice_score(confusion):
    return 2 * confusion[1][1] / (2 * confusion[1][1] + confusion[1][0] + confusion[0][1])

def conf_matrix(segmentation, ground_truth):
    s = np.copy(segmentation)
    gt = np.copy(ground_truth)

    C = confusion_matrix(gt, s)
    return C

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
    return dice_score(C)

if len(sys.argv) != 3:
    print("Please provide the segmentation dir and the ground truth dir")

seg_dir = sys.argv[1]
gt_dir = sys.argv[2]

dices = [[],[],[]]
totalC = None
for segmentation, gt in zip(sorted(os.listdir(seg_dir)), sorted(os.listdir(gt_dir))):

    segmentation = os.path.join(seg_dir, segmentation)
    gt = os.path.join(gt_dir, gt)
    
    segmentation = sitk.GetArrayFromImage(sitk.ReadImage(segmentation))
    ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(gt))

    segmentation = segmentation.reshape((-1,))
    ground_truth = ground_truth.reshape((-1,))

    C = conf_matrix(ground_truth, segmentation)
    if totalC is None:
        totalC = C
    else:
        totalC = totalC + C
    print(C)
    
    dice = dice_score(C)
    dices[0].append(evaluate_region(segmentation, ground_truth, [1,2,3,4], [0]))
    dices[1].append(evaluate_region(segmentation, ground_truth, [1,3,4], [0,2]))
    dices[2].append(evaluate_region(segmentation, ground_truth, [4], [0,1,2,3]))

print(dices)
for dices in dices:
    print(sum(dices)/len(dices))
print(totalC)

for row in totalC:
    s = sum(row)
    for x in row:
        print("{:1.4f}%".format(x/s * 100), end=' ')
    print()

#evaluate_region(segmentation, ground_truth, [1,2,3,4], [0])
#evaluate_region(segmentation, ground_truth, [1,3,4], [0,2])
#evaluate_region(segmentation, ground_truth, [4], [0,1,2,3])
