#script to parse results from the online platform

import sys
import csv
import re
import math
import numpy as np
import matplotlib.pyplot as plt

def std_dev(scores, means):
    stds = {}
    for region in scores:
        variance = 0
        for score in scores[region]:
            variance += (score - means[region])**2
        stds[region] = math.sqrt(variance / (len(scores[region]) - 1))
    return stds


assert(len(sys.argv) == 2)
with open(sys.argv[1], newline='') as csvfile:
    results = list(csv.reader(csvfile, delimiter=';'))

print(results[0])
print(results[1])
print(results[2])
print(results[3])
print(results[4])
headers = results[4]

results = [r for r in results if len(r) > 1 and r[0] == 'Sebastian']

submissions = {}

#sort results by submission
pattern = 'sub(\d+)'
for row in results:
    matches = re.findall(pattern, row[headers.index('Name')])
    if len(matches) != 1:
        raise Exception("Error while processing the CSV. The following row didn't match for a single sub number", row)
    sub_number = int(matches[0]) 
    if sub_number in submissions:
        submissions[sub_number].append(row)
    else:
        submissions[sub_number] = [row]


#sort regions by scan
scan_pattern = '03(\d{2})'
for k in submissions:
    scans = {}
    for i in range(1,11):
        scans[i] = []
    for row in submissions[k]:
        matches = re.findall(scan_pattern, row[headers.index('Name')])
        if len(matches) != 1:
            raise Exception("Error while processing the CSV. Row doesn't contain a submission number", row)
        scan_number = int(matches[0])
        scans[scan_number].append(row)
    if len(scans) != 10:
        print("Submission {} doesn't have 10 scans".format(k))
    else:
        submissions[k] = scans

#sort submissions by region
print(headers)
for h in headers:
    print(h)
region_column = headers.index('Region')

for k in submissions:
    for scan in submissions[k]:
        if len(submissions[k][scan]) != 3:
            print("Submission {} contains duplicates for scan {}".format(k, scan))
        regions = {}
        for row in submissions[k][scan]:
            region = int(row[region_column])
            regions[region] = row
        submissions[k][scan] = regions

dice_mean = {}
dice_std = {}
sens_mean = {}
sens_std = {}
ppv_mean = {}
ppv_std = {}

#compute average dice scores and std for submissions
for sub in submissions:
    dice_scores = {1:[], 2:[], 3:[]}
    sens_scores = {1:[], 2:[], 3:[]}
    ppv_scores = {1:[], 2:[], 3:[]}

    scans = submissions[sub]
    for scan in scans:
        for region in scans[scan]:
            dice_scores[region].append(float(scans[scan][region][headers.index('Dice')]))
            sens_scores[region].append(float(scans[scan][region][headers.index('Sensitivity')]))
            ppv_scores[region].append(float(scans[scan][region][headers.index('Positive Predictive Value')]))
    
    dice_mean[sub] = {}
    sens_mean[sub] = {}
    ppv_mean[sub] = {}
    for region in scans[scan]:
        dice_mean[sub][region] = sum(dice_scores[region])/len(dice_scores[region])
        sens_mean[sub][region] = sum(sens_scores[region])/len(sens_scores[region])
        ppv_mean[sub][region] = sum(ppv_scores[region])/len(ppv_scores[region])
    
    dice_std[sub] = std_dev(dice_scores, dice_mean[sub])
    sens_std[sub] = std_dev(sens_scores, sens_mean[sub])
    ppv_std[sub] = std_dev(ppv_scores, ppv_mean[sub])

dices = [[],[],[]]
#report values
for sub in dice_mean:
    mean = dice_mean[sub]
    std = dice_std[sub]
    print("Sub {:2}, {:0.4f}±{:0.4f} {:0.4f}±{:0.4f} {:0.4f}±{:0.4f}" .format(sub, mean[1], std[1], mean[2], std[2], mean[3], std[3]))
    for region in [1,2,3]:
        dices[region-1].append(mean[region])

maxs = [dices[region].index(max(dices[region])) for region in range(3)]
print("Maximums at {}".format(maxs))

def plot_results(sub):
    scores = [[],[],[]]
    for scan in submissions[sub]:
        for k, region in enumerate(submissions[sub][scan]):
           scores[k].append(float(submissions[sub][scan][region][headers.index('Dice')]))
    plt.figure(1)
    plt.boxplot(scores, sym='', showmeans=True)
    plt.xticks([1, 2, 3], ['Complete', 'Core', 'Enhancing'])
    plt.title("Box plot showing the submission results for the challenge dataset")

    plt.figure(2)
    means = np.array(list(dice_mean[sub].values()))
    stds = np.array(list(dice_std[sub].values()))
    mins = np.array([min(r) for r in scores])
    maxs = np.array([max(r) for r in scores])
    plt.errorbar([1,2,3], means, yerr=stds, fmt='o')
    plt.plot([1,2,3], mins, 'rx')
    plt.plot([1,2,3], maxs, 'rx')
    plt.xlim((0,4))
    plt.ylim((0,1))
    plt.xticks([1, 2, 3], ['Complete', 'Core', 'Enhancing'])
    plt.show()

def show_results(sub):
    print("Showing individual scans for submissions {}".format(sub))
    print("Scan\tRegion\tDice\tSensitivity\tPPV")
    dices = [[],[],[]]
    for region in [1,2,3]:
        for scan in submissions[sub]:
            dice = float(submissions[sub][scan][region][headers.index('Dice')])
            dices[region-1].append(dice)
            sensitivity = float(submissions[sub][scan][region][headers.index('Sensitivity')])
            ppv = float(submissions[sub][scan][region][headers.index('Positive Predictive Value')])
            print("{}\t{}\t{:0.3f}\t{:0.3f}\t\t{:0.3f}".format(scan, region, dice, sensitivity, ppv))
        print("{:0.4f}±{:0.4f}\t{:0.4f}±{:0.4f}\t{:0.4f}±{:0.4f}" .format(dice_mean[sub][region], dice_std[sub][region], sens_mean[sub][region], sens_std[sub][region], ppv_mean[sub][region], ppv_std[sub][region]))
    xs = [i for i in range(1,11)]
    plt.plot(xs, dices[0], 'x', xs, dices[1], 'o', xs, dices[2], 'x')
    plt.xlim(0, 11)
    plt.ylim(0,1)
    plt.show()
#show_results(57)
#show_results(60)
plot_results(60)
