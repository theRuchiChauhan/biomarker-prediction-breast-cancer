"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Code for statistical Analysis using Mann-Whitney U test
Reference: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
Author: Ruchi Chauhan
Date: 6 August 2020
-------------------------------------------------------------------------------------
(*) H0: null hypothesis is that there is no difference between the distributions of the data samples
(*) For the test to be effective, it requires at least 20 observations in each data sample
-------------------------------------------------------------------------------------
python3 MannWhitney.py noTP53feats_CancerVsNormalExternalSet_L0.csv TP53feats_CancerVsNormalExternalSet_L0.csv TP53
>> TP53

python3 MannWhitney.py TP53
"""
import pandas as pd
import sys
from scipy.stats import mannwhitneyu

task = sys.argv[1]
filePath1 = task+"_Neg_Disc_feats.csv"
filePath2 = task+"_Pos_Disc_feats.csv"

data1 = pd.read_csv(filePath1,header=None)
data2 = pd.read_csv(filePath2,header=None)

statsList = [] ; pList = [] ; resultList = []
############# Create Labels ################
labelList = []
#ListMajor = ['Mean','STD','Skewness','Kurtosis','Entropy']
#ListMinor = ['Area','MajorAxis','MinorAxis','AxesRatio','mean_R','mean_G','mean_B','Mean Distance','Max Distance','Min Distance'] 
## Check these 🚩

#for x in ListMajor:
#    for y in ListMinor:
#        labelList.append(x +'_'+ y)
labelList = ['Mean_Area', 'Mean_MajorAxis', 'Mean_MinorAxis', 'Mean_AxesRatio', 'Mean_mean_R', 'Mean_mean_G', 'Mean_mean_B', 'Mean_Mean Distance', 'Mean_Max Distance', 'Mean_Min Distance', 'STD_Area', 'STD_MajorAxis', 'STD_MinorAxis', 'STD_AxesRatio', 'STD_mean_R', 'STD_mean_G', 'STD_mean_B', 'STD_Mean Distance', 'STD_Max Distance', 'STD_Min Distance', 'Skewness_Area', 'Skewness_MajorAxis', 'Skewness_MinorAxis', 'Skewness_AxesRatio', 'Skewness_mean_R', 'Skewness_mean_G', 'Skewness_mean_B', 'Skewness_Mean Distance', 'Skewness_Max Distance', 'Skewness_Min Distance', 'Kurtosis_Area', 'Kurtosis_MajorAxis', 'Kurtosis_MinorAxis', 'Kurtosis_AxesRatio', 'Kurtosis_mean_R', 'Kurtosis_mean_G', 'Kurtosis_mean_B', 'Kurtosis_Mean Distance', 'Kurtosis_Max Distance', 'Kurtosis_Min Distance', 'Entropy_Area', 'Entropy_MajorAxis', 'Entropy_MinorAxis', 'Entropy_AxesRatio', 'Entropy_mean_R', 'Entropy_mean_G', 'Entropy_mean_B', 'Entropy_Mean Distance', 'Entropy_Max Distance', 'Entropy_Min Distance', 'Shape.FSD1', 'Shape.FSD2', 'Shape.FSD3', 'Shape.FSD4', 'Shape.FSD5', 'Shape.FSD6', 'Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness', 'Gradient.Mag.Kurtosis', 'Gradient.Mag.HistEntropy', 'Gradient.Mag.HistEnergy', 'Gradient.Canny.Sum', 'Gradient.Canny.Mean', 'Haralick.ASM.Mean', 'Haralick.ASM.Range', 'Haralick.Contrast.Mean', 'Haralick.Contrast.Range', 'Haralick.Correlation.Mean', 'Haralick.Correlation.Range', 'Haralick.SumOfSquares.Mean', 'Haralick.SumOfSquares.Range', 'Haralick.IDM.Mean', 'Haralick.IDM.Range', 'Haralick.SumAverage.Mean', 'Haralick.SumAverage.Range', 'Haralick.SumVariance.Mean', 'Haralick.SumVariance.Range', 'Haralick.SumEntropy.Mean', 'Haralick.SumEntropy.Range', 'Haralick.Entropy.Mean', 'Haralick.Entropy.Range', 'Haralick.DifferenceVariance.Mean', 'Haralick.DifferenceVariance.Range', 'Haralick.DifferenceEntropy.Mean', 'Haralick.DifferenceEntropy.Range', 'Haralick.IMC1.Mean', 'Haralick.IMC1.Range', 'Haralick.IMC2.Mean', 'Haralick.IMC2.Range', 'Size.Area', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Size.Perimeter', 'Shape.Circularity', 'Shape.Eccentricity', 'Shape.EquivalentDiameter', 'Shape.Extent', 'Shape.MinorMajorAxisRatio', 'Shape.Solidity']
failCount = 0
############# Calculate Stats ##############
for i in range(1, data1.shape[1]): ## Iterating over columns
# for i in range(51, 91): ## Iterating over columns
    list1 = data1.iloc[:,i].values
    list2 = data2.iloc[:,i].values
    try:
        stat, p = mannwhitneyu(list1, list2)
    except ValueError:
        # import pdb; pdb.set_trace()
        print('Defect in feature number '+str(i))
        continue
    statsList.append(stat)
    pList.append(p)

    # print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        # print('SAME (fail to reject H0)')
        resultList.append('Same')
        failCount +=1
    else:
#        print('Different distribution (reject H0)')
        resultList.append('Different')
print(f"{failCount} features are same")
############# Print Stats to a file ####### 
import csv


rows = zip(labelList,pList,resultList)
## use this is you want sorted list
# rows = sorted(zip(labelList,pList,resultList),key=lambda x: x[1])
with open(task+'_stats_100feats_unsorted.csv', 'w') as f:
    wr = csv.writer(f)
    for row in rows:
        wr.writerow(row)
