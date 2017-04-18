import numpy as np
import sys
import SimpleITK as sitk

print(sys.argv)

if len(sys.argv) != 3:
    print("Please provide the segmented image and the ground truth images")

segmentation = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[1]))
ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[2]))

print("Segmentation: ", segmentation.shape)
print("Ground truth: ", ground_truth.shape)

def dice_score(TP, FP, FN):
    return 2.0 * TP / (FP + 2.0*TP + FN) 

#positive predictive value
def ppv(TP, FP):
    return float(TP) / (TP + FP)

def sensitivity(TP, FN):
    return float(TP) / (TP + FN)

def calculate_dice_score(segmentation, ground_truth):
    true_positives = [0,0,0]
    false_positives = [0,0,0]
    false_negatives = [0,0,0]
    true_negatives = [0,0,0]
    
    metrics = ["complete", "core", "enhancing"]
    positives = [[1,2,3,4],[1,3,4],[4]]
    for slice_number in range(segmentation.shape[0]):
        print("Slice", slice_number)
        for y in range(segmentation.shape[1]):
            for x in range(segmentation.shape[2]):
                s_val = segmentation[slice_number, y, x]
                gt_val = ground_truth[slice_number, y, x]
                for (i, positive) in enumerate(positives):
                    if s_val in positive and gt_val in positive:
                        true_positives[i] += 1
                    if s_val in positive and gt_val not in positive:
                        false_positives[i] += 1
                    if s_val not in positive and gt_val in positive:
                        false_negatives[i] += 1
                    if s_val not in positive and gt_val not in positive:
                        true_negatives[i] += 1

    print("TP", true_positives)
    print("FP", false_positives)
    print("FN", false_negatives)
    print("TN", true_negatives)
    
    for (i, metric) in enumerate(metrics):
        DSC = dice_score(true_positives[i], false_positives[i], false_negatives[i])
        PPV = ppv(true_positives[i], false_positives[i])
        S = sensitivity(true_positives[i], false_negatives[i])
        print("Dice score", metric,  DSC)
        print("PPV score", metric,  PPV)
        print("Sensitivity score", metric,  S)
    
    return dice_score

calculate_dice_score(segmentation, ground_truth)


