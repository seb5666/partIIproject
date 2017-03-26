#script to parse results from the online platform

import sys
import csv
import re
import math
import numpy as np
import matplotlib.pyplot as plt

assert(len(sys.argv) == 2)

with open(sys.argv[1], newline='') as csvfile:
    results = list(csv.reader(csvfile, delimiter=';'))

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
#compute average dice scores and std for submissions
for sub in submissions:
    dice_scores = {1:0, 2:0, 3:0}
    for scan in submissions[sub]:
        for region in submissions[sub][scan]:
            dice_scores[region] += float(submissions[sub][scan][region][headers.index('Dice')]) / 10
    dice_mean[sub] = dice_scores

    scores = {1:[], 2:[], 3:[]}
    scans = submissions[sub]
    for scan in scans:
        for region in scans[scan]:
            scores[region].append(float(scans[scan][region][headers.index('Dice')]))
    std = {1:0, 2:0, 3:0}
    for region in scores:
        variance = 0
        for score in scores[region]:
            variance += (score - dice_mean[sub][region])**2
        std[region] = math.sqrt(variance / 9)
    dice_std[sub] = std

#report values
for sub in dice_mean:
    mean = dice_mean[sub]
    std = dice_std[sub]
    print("Sub {:2}, {:0.3f}±{:0.3f} {:0.3f}±{:0.3f} {:0.3f}±{:0.3f}" .format(sub, mean[1], std[1], mean[2], std[2], mean[3], std[3]))

def plot_results(sub):
    scores = [[],[],[]]
    for scan in submissions[sub]:
        for k, region in enumerate(submissions[sub][scan]):
           scores[k].append(float(submissions[sub][scan][region][headers.index('Dice')]))
    print(scores)
    plt.boxplot(scores, sym='')
    plt.xticks([1, 2, 3], ['Complete', 'Core', 'Enhancing'])
    plt.title("Box plot showing the submission results for the challenge dataset")
    plt.show()
plot_results(32)
