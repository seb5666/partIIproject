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
def calculate_dice_score(segmentation, ground_truth):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    positive = [1,2,3,4]
    for slice_number in range(segmentation.shape[0]):
        print("Slice", slice_number)
        for y in range(segmentation.shape[1]):
            for x in range(segmentation.shape[2]):
                s_val = segmentation[slice_number, y, x]
                gt_val = ground_truth[slice_number, y, x]
                if s_val in positive and gt_val in positive:
                    true_positives += 1
                if s_val in positive and gt_val not in positive:
                    false_positives += 1
                if s_val not in positive and gt_val in positive:
                    false_negatives += 1

    print("TP", true_positives)
    print("FP", false_positives)
    print("FN", false_negatives)

    DSC = dice_score(true_positives, false_positives, false_negatives)
    print("Dice score", DSC)
    
    return dice_score

calculate_dice_score(segmentation, ground_truth)


