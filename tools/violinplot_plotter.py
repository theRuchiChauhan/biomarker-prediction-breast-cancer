"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Author: Ruchi Chauhan; Date: 5 Sept 2020
Violin Plots for characterstic features as per Value importance of Random Forest

Create a structure like this:
AttributionValues featureName label   
1.2                 majAxs      neg
2.3                 majAxs      neg
43.2                majAxs      pos
2.8                 majAxs      pos   
3.7                 majAxs      pos
25.9                area        neg
61.5                area        neg
0.25                area        pos
40.2                area        pos

Enter task and list of features
>> Handles the flip labelling of ER & PR

python3 violinPlotPlotter.py ER ['Mean_Area','Mean_AxesRatio', 'Mean_mean_B']
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

task = sys.argv[1]
# impList = sys.argv[2].split(',')
#task = 'ER'
if task == 'ER' or task == 'PR':
    file1 = task+'_Neg_Disc_feats.csv'
    file2 = task+'_Pos_Disc_feats.csv'
else: 
    file1 = task+'_Pos_Disc_feats.csv'
    file2 = task+'_Neg_Disc_feats.csv'
    
posData = pd.read_csv(file1,header=None)
negData = pd.read_csv(file2,header=None)

bigFeatureList = ['Mean_Area', 'Mean_MajorAxis', 'Mean_MinorAxis', 'Mean_AxesRatio', 'Mean_mean_R', 'Mean_mean_G', 'Mean_mean_B', 'Mean_Mean Distance', 'Mean_Max Distance', 'Mean_Min Distance', 'STD_Area', 'STD_MajorAxis', 'STD_MinorAxis', 'STD_AxesRatio', 'STD_mean_R', 'STD_mean_G', 'STD_mean_B', 'STD_Mean Distance', 'STD_Max Distance', 'STD_Min Distance', 'Skewness_Area', 'Skewness_MajorAxis', 'Skewness_MinorAxis', 'Skewness_AxesRatio', 'Skewness_mean_R', 'Skewness_mean_G', 'Skewness_mean_B', 'Skewness_Mean Distance', 'Skewness_Max Distance', 'Skewness_Min Distance', 'Kurtosis_Area', 'Kurtosis_MajorAxis', 'Kurtosis_MinorAxis', 'Kurtosis_AxesRatio', 'Kurtosis_mean_R', 'Kurtosis_mean_G', 'Kurtosis_mean_B', 'Kurtosis_Mean Distance', 'Kurtosis_Max Distance', 'Kurtosis_Min Distance', 'Entropy_Area', 'Entropy_MajorAxis', 'Entropy_MinorAxis', 'Entropy_AxesRatio', 'Entropy_mean_R', 'Entropy_mean_G', 'Entropy_mean_B', 'Entropy_Mean Distance', 'Entropy_Max Distance', 'Entropy_Min Distance', 'Shape.FSD1', 'Shape.FSD2', 'Shape.FSD3', 'Shape.FSD4', 'Shape.FSD5', 'Shape.FSD6', 'Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness', 'Gradient.Mag.Kurtosis', 'Gradient.Mag.HistEntropy', 'Gradient.Mag.HistEnergy', 'Gradient.Canny.Sum', 'Gradient.Canny.Mean', 'Haralick.ASM.Mean', 'Haralick.ASM.Range', 'Haralick.Contrast.Mean', 'Haralick.Contrast.Range', 'Haralick.Correlation.Mean', 'Haralick.Correlation.Range', 'Haralick.SumOfSquares.Mean', 'Haralick.SumOfSquares.Range', 'Haralick.IDM.Mean', 'Haralick.IDM.Range', 'Haralick.SumAverage.Mean', 'Haralick.SumAverage.Range', 'Haralick.SumVariance.Mean', 'Haralick.SumVariance.Range', 'Haralick.SumEntropy.Mean', 'Haralick.SumEntropy.Range', 'Haralick.Entropy.Mean', 'Haralick.Entropy.Range', 'Haralick.DifferenceVariance.Mean', 'Haralick.DifferenceVariance.Range', 'Haralick.DifferenceEntropy.Mean', 'Haralick.DifferenceEntropy.Range', 'Haralick.IMC1.Mean', 'Haralick.IMC1.Range', 'Haralick.IMC2.Mean', 'Haralick.IMC2.Range', 'Size.Area', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Size.Perimeter', 'Shape.Circularity', 'Shape.Eccentricity', 'Shape.EquivalentDiameter', 'Shape.Extent', 'Shape.MinorMajorAxisRatio', 'Shape.Solidity']

valueList = []; featureList = []; labelList = []; indexList = []

impList = bigFeatureList[:50]
indexList = [bigFeatureList.index(imp) for imp in impList]

if (task == 'TP53' or task == 'PIK3CA'):   
    legendLabelPos = task
    legendLabelNeg = '~ '+task
else:
    legendLabelPos = task+' +ive'
    legendLabelNeg = task+' -ive'
figCount = 1
plt_count = 0
### create structure for plotting
for j, index in enumerate(indexList):

    singleValueList = negData.iloc[:,index].tolist() + posData.iloc[:,index].tolist()
    # singleValueList = preprocessing.normalize(np.array(singleValueList).reshape((-1,1)),axis=0).flatten().tolist() ## normalise

    valueList = singleValueList
    featureList = [impList[j] for i in range(negData.shape[0]+posData.shape[0])]
    
    labelList = [legendLabelPos for i in range(posData.shape[0])] + [legendLabelNeg for i in range(negData.shape[0])]

    customData = {'AttributeValues':valueList,'featureName':featureList,'label':labelList}
    customDatadf = pd.DataFrame(customData,columns=['AttributeValues','featureName','label'])

# customDatadf.loc['AttributeValues'] = (customDatadf.loc['AttributeValues'] - customDatadf.loc['AttributeValues'].mean()) / customDatadf.loc['AttributeValues'].std()
### Plot
    if (j+1) % 5 == 0:
#        import pdb ; pdb.set_trace()
        plt_count = 0
        if j != 0:
            plt.show()
            plt.savefig('sample_'+task+'_'+str(figCount)+'.png')
            figCount += 1
            plt.figure()
    plt.subplot(1, 5, plt_count+1)
    plt_count += 1
    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    sns.violinplot(x="featureName", y="AttributeValues", hue="label",
                   split=True, inner="quart",
                   palette={legendLabelNeg: "r", legendLabelPos: "b"},
                   data=customDatadf)
    if j != 0:
        plt.ylabel('')
    plt.xlabel('')
# plt.xlabel('featureName')
# plt.title('Distributions for '+task)    
# plt.savefig('sample_'+task+'.png')
# plt.show()

################### PLAYGROUND ##################
# posData.shape
# (5759, 100)
# negData.shape
# (3823, 100)
# customDatadf = (customDatadf - customDatadf.mean()) / customDatadf.std()
############################################### WORKING FOR SINGLE PLOT ######################################
# for j, index in enumerate(indexList):

#     singleValueList = negData.iloc[:,index].tolist() + posData.iloc[:,index].tolist()
#     # singleValueList = preprocessing.normalize(np.array(singleValueList).reshape((-1,1)),axis=0).flatten().tolist() ## normalise

#     valueList = valueList + singleValueList
#     featureList = featureList + [impList[j] for i in range(negData.shape[0]+posData.shape[0])]
#     labelList = labelList + ['neg' for i in range(posData.shape[0])] + ['pos' for i in range(negData.shape[0])]

# customData = {'AttributeValues':valueList,'featureName':featureList,'label':labelList}
# customDatadf = pd.DataFrame(customData,columns=['AttributeValues','featureName','label'])

# # customDatadf.loc['AttributeValues'] = (customDatadf.loc['AttributeValues'] - customDatadf.loc['AttributeValues'].mean()) / customDatadf.loc['AttributeValues'].std()
# ### Plot
# sns.set(style="whitegrid", palette="pastel", color_codes=True)
# sns.violinplot(x="featureName", y="AttributeValues", hue="label",
#                 split=True, inner="quart",
#                 palette={"neg": "y", "pos": "b"},
#                 data=customDatadf)
# plt.savefig('sample_'+task+'.png')
# plt.show()
