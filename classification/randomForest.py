"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Random Forest
Author: Ruchi Chauhan
Date: 8Aug'20
Reference: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

(*) Classify using random forest, print tree, print variable importance ☑
(*) Convert code to take only morphological or only spatial attributes ☑
(*) Allow to include features as per pValues 🔲
(*) do 5 fold cross validation
---------------------------------------------------------------------------------------
## negative first; select task 🚩
python3 randomForest.py notp53_patches_L0_morph_feat.csv tp53_patches_L0_morph_feat.csv 
"""

import sys
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.model_selection import GridSearchCV

from sklearn.tree import export_graphviz
import pydot

import matplotlib.pyplot as plt


def get_metrics_rf(predictions, probs, labels):
    errors = abs(predictions - labels)
    # print('Mean Absolute Error:', round(np.mean(errors), 2))
    accuracy = np.sum(predictions == labels)/(predictions.shape[0])
    f1Score = f1_score(labels, predictions)
    auroc = roc_auc_score(labels, probs[:, 1])
    precision, recall, thresholds = pr_curve(labels, probs[:,1])
    auprc = auc(recall, precision)
    return errors, accuracy, f1Score, auroc, auprc

def aggregate_metrics(acc_list, f1_list, auroc_list, auprc_list, mse_list):
    final_acc = np.mean(np.array(acc_list))
    final_f1 = np.mean(np.array(f1_list))
    final_mse = np.mean(np.array(mse_list))
    final_auroc = np.mean(np.array(auroc_list))
    final_auprc = np.mean(np.array(auprc_list))
    return final_acc, final_f1, final_mse, final_auroc, final_auprc

def get_optimal_params(X, y):
    rf = RandomForestClassifier(random_state = 42, n_jobs=-1)
    gs = GridSearchCV(rf, {'n_estimators': list(range(10,100,20)),\
                           'max_depth': list(range(10,50,10))+[None]},
                      cv=3)
    gs.fit(X, y)
    return gs
##################### Prepare Data ########################


biomarker = sys.argv[1]
# file2 = biomarker+'_allFeats_CancerVsNormalExternalSet_L0.csv'
# file1 = 'no'+biomarker+'_allFeats_CancerVsNormalExternalSet_L0.csv'
# file2 = biomarker+'_Neg_Disc_feats.csv' # 🚨
# file1 = biomarker+'_Pos_Disc_feats.csv'
# ## The file with more samples will be negative
if biomarker == 'PR':
    file1 = biomarker+'_Neg_Disc_feats.csv'
    file2 = biomarker+'_Pos_Disc_feats.csv'
else:
    file1 = biomarker+'_Pos_Disc_feats.csv'
    file2 = biomarker+'_Neg_Disc_feats.csv'
#file1 = sys.argv[1]
#file2 = sys.argv[2]
# statFile = sys.argv[3]
data1 = pd.read_csv(file1,header=None).dropna()
# data1 = data1.drop([0],axis=1) ## removingthe patchnames
# data1 = pd.DataFrame(data1.values)
data2 = pd.read_csv(file2,header=None).dropna()
# data2 = data2.drop([0],axis=1)
# data2 = pd.DataFrame(data2.values)
data2 = data2.sample(n=data1.shape[0])

# taskName = biomarker+'_'+task
# task =  'intensityFeatures'#'both'#'spatialFeatures'#'nucleiFeatures'#pValues	🚩
task = sys.argv[2]#'both'

# data2 = data2.iloc[:data1.shape[0], :]

true1 = np.zeros(data1.shape[0])
# true2 = np.ones(data2.shape[0])
true2 = np.ones(data1.shape[0])
#print(data2.shape)
#print(data1.shape)

X = pd.concat([data1,data2])
y = np.concatenate((true1,true2),axis = 0)

X_normed = [(X.iloc[:, i] - min(X.iloc[:, i]))/max(X.iloc[:, i]) for i in range(X.shape[1])]
X = pd.concat(X_normed, 1)
# X.values

###################### Get Labels #########################
labels = ['Mean_Area', 'Mean_MajorAxis', 'Mean_MinorAxis', 'Mean_AxesRatio', 'Mean_mean_R', 'Mean_mean_G', 'Mean_mean_B', 'Mean_Mean Distance', 'Mean_Max Distance', 'Mean_Min Distance', 'STD_Area', 'STD_MajorAxis', 'STD_MinorAxis', 'STD_AxesRatio', 'STD_mean_R', 'STD_mean_G', 'STD_mean_B', 'STD_Mean Distance', 'STD_Max Distance', 'STD_Min Distance', 'Skewness_Area', 'Skewness_MajorAxis', 'Skewness_MinorAxis', 'Skewness_AxesRatio', 'Skewness_mean_R', 'Skewness_mean_G', 'Skewness_mean_B', 'Skewness_Mean Distance', 'Skewness_Max Distance', 'Skewness_Min Distance', 'Kurtosis_Area', 'Kurtosis_MajorAxis', 'Kurtosis_MinorAxis', 'Kurtosis_AxesRatio', 'Kurtosis_mean_R', 'Kurtosis_mean_G', 'Kurtosis_mean_B', 'Kurtosis_Mean Distance', 'Kurtosis_Max Distance', 'Kurtosis_Min Distance', 'Entropy_Area', 'Entropy_MajorAxis', 'Entropy_MinorAxis', 'Entropy_AxesRatio', 'Entropy_mean_R', 'Entropy_mean_G', 'Entropy_mean_B', 'Entropy_Mean Distance', 'Entropy_Max Distance', 'Entropy_Min Distance', 'Shape.FSD1', 'Shape.FSD2', 'Shape.FSD3', 'Shape.FSD4', 'Shape.FSD5', 'Shape.FSD6', 'Gradient.Mag.Mean', 'Gradient.Mag.Std', 'Gradient.Mag.Skewness', 'Gradient.Mag.Kurtosis', 'Gradient.Mag.HistEntropy', 'Gradient.Mag.HistEnergy', 'Gradient.Canny.Sum', 'Gradient.Canny.Mean', 'Haralick.ASM.Mean', 'Haralick.ASM.Range', 'Haralick.Contrast.Mean', 'Haralick.Contrast.Range', 'Haralick.Correlation.Mean', 'Haralick.Correlation.Range', 'Haralick.SumOfSquares.Mean', 'Haralick.SumOfSquares.Range', 'Haralick.IDM.Mean', 'Haralick.IDM.Range', 'Haralick.SumAverage.Mean', 'Haralick.SumAverage.Range', 'Haralick.SumVariance.Mean', 'Haralick.SumVariance.Range', 'Haralick.SumEntropy.Mean', 'Haralick.SumEntropy.Range', 'Haralick.Entropy.Mean', 'Haralick.Entropy.Range', 'Haralick.DifferenceVariance.Mean', 'Haralick.DifferenceVariance.Range', 'Haralick.DifferenceEntropy.Mean', 'Haralick.DifferenceEntropy.Range', 'Haralick.IMC1.Mean', 'Haralick.IMC1.Range', 'Haralick.IMC2.Mean', 'Haralick.IMC2.Range', 'Size.Area', 'Size.MajorAxisLength', 'Size.MinorAxisLength', 'Size.Perimeter', 'Shape.Circularity', 'Shape.Eccentricity', 'Shape.EquivalentDiameter', 'Shape.Extent', 'Shape.MinorMajorAxisRatio', 'Shape.Solidity']

extraFeatureIdx = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 97, 98]




## for testing only on nuclei or spatial features
# nucleiFeatureIdx = [0,1,2,3,4,5,6,10,11,12,13,14,15,16,20,21,22,23,24,25,26,30,31,32,33,34,35,36,40,41,42,43,44,45,46]
nonSpatialFeatureIdx = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# nonSpatialFeatureIdx = [num+1 for num in nonSpatialFeatureIdx]
spatialFeatureIdx = list(set(range(100)) - set(nonSpatialFeatureIdx))
# spatialFeatureIdx = list(set(range(1, 101)) - set(nonSpatialFeatureIdx))

noIntensityFeatureIdx = [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
# noIntensityFeatureIdx = [num+1 for num in noIntensityFeatureIdx]
intensityFeatureIdx = list(set(range(100)) - set(noIntensityFeatureIdx))
# intensityFeatureIdx = list(set(range(1, 101)) - set(noIntensityFeatureIdx))

noNucleiFeatureIdx = spatialFeatureIdx+intensityFeatureIdx
nucleiFeatureIdx = list(set(range(100)) - set(noNucleiFeatureIdx))

if task == 'nucleiFeatures':
    X.drop((extraFeatureIdx+noNucleiFeatureIdx), axis=1, inplace=True)
    labels = np.array(labels)[nucleiFeatureIdx]
elif task == 'spatialFeatures':
    X.drop((extraFeatureIdx+nonSpatialFeatureIdx), axis=1,inplace=True)
    labels = np.array(labels)[spatialFeatureIdx]
elif task == 'intensityFeatures':
    X.drop((extraFeatureIdx+noIntensityFeatureIdx), axis=1,inplace=True)
    labels = np.array(labels)[intensityFeatureIdx]
# if task == 'pValues':
elif task == 'nuclei+spatial':
    X.drop((extraFeatureIdx+intensityFeatureIdx), axis=1, inplace=True)
    labels = np.array(labels)[nucleiFeatureIdx+spatialFeatureIdx]
elif task == 'intensity+spatial':
    X.drop((extraFeatureIdx+nucleiFeatureIdx), axis=1, inplace=True)
    labels = np.array(labels)[intensityFeatureIdx+spatialFeatureIdx]
elif task == 'intensity+nuclei':
    X.drop((extraFeatureIdx+spatialFeatureIdx), axis=1, inplace=True)
    labels = np.array(labels)[intensityFeatureIdx+nucleiFeatureIdx]
# ## Remember that list indexes at 0. Our Idxlist is also indexed at 0, so no issues  
# 	for index in sorted(sortedIdxList, reverse=True): ## reverse sort the list to prevent reindexing
# 	...     del labels[index]    #🚨 but don't delete from sortedIdxList. delete the removeIdxList    
# #print(labels)            
X = X.dropna(1)
################### Classify ##############################
# train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.20, random_state = 42)
# pdb.set_trace()
# grid_search_obj = get_optimal_params(X, y)
skf = StratifiedKFold(10, random_state=42)

acc_pw_list_trn = [] ; f1_pw_list_trn = [] ; mse_pw_list_trn = []
auroc_list_trn = [] ; auprc_list_trn = []
acc_pw_list_tst = [] ; f1_pw_list_tst = [] ; mse_pw_list_tst = []
auroc_list_tst = [] ; auprc_list_tst = []
split_id = 0
X_np = X.values
for trn_idx, tst_idx in skf.split(X_np, y):
    t1 = time.time()
    train_features = X_np[trn_idx]
    test_features = X_np[tst_idx]
    train_labels = y[trn_idx]
    test_labels = y[tst_idx]
    rf = RandomForestClassifier(n_estimators = 90, max_depth=20,
                                criterion='entropy', random_state = 42,
                                n_jobs=-1)
    # old n_estimators = 80
    rf.fit(train_features, train_labels);
    probs_tst = rf.predict_proba(test_features)
    preds_tst = rf.predict(test_features)
    probs_trn = rf.predict_proba(train_features)
    preds_trn = rf.predict(train_features)
    errors_trn, accuracy_trn, f1Score_trn,\
            auroc_trn, auprc_trn = get_metrics_rf(preds_trn, probs_trn,
                                                  train_labels)
    errors_tst, accuracy_tst, f1Score_tst,\
            auroc_tst, auprc_tst = get_metrics_rf(preds_tst, probs_tst,
                                                  test_labels)
    # print('Split num. '+str(split_id)+': Acc-'+str(accuracy)
    #           + ' F1-'+str(f1Score))
    print('Split num. %d : train -- Acc- %.3f ; F1- %.3f ; AUROC- %.3f ;'
          'AUPRC- %.3f' %(split_id, accuracy_trn, f1Score_trn, auroc_trn,
                          auprc_trn))
    print('Split num. %d : test -- Acc- %.3f ; F1- %.3f ; AUROC- %.3f ;'
          'AUPRC- %.3f' %(split_id, accuracy_tst, f1Score_tst, auroc_tst,
                          auprc_tst))
    split_id += 1
    print('Took '+str(time.time() - t1)+' seconds for one fold')
    acc_pw_list_trn.append(accuracy_trn)
    acc_pw_list_tst.append(accuracy_tst)
    f1_pw_list_trn.append(f1Score_trn)
    f1_pw_list_tst.append(f1Score_tst)
    auroc_list_trn.append(auroc_trn)
    auroc_list_tst.append(auroc_tst)
    auprc_list_trn.append(auprc_trn)
    auprc_list_tst.append(auprc_tst)
    mse_pw_list_trn.append(np.mean(errors_trn))
    mse_pw_list_tst.append(np.mean(errors_tst))

final_acc_tst, final_f1_tst, final_mse_tst, final_auroc_tst,\
        final_auprc_tst = aggregate_metrics(acc_pw_list_tst, f1_pw_list_tst,
                                            auroc_list_tst, auprc_list_tst,
                                            mse_pw_list_tst)
final_acc_trn, final_f1_trn, final_mse_trn, final_auroc_trn,\
        final_auprc_trn = aggregate_metrics(acc_pw_list_trn, f1_pw_list_trn,
                                            auroc_list_trn, auprc_list_trn,
                                            mse_pw_list_trn)
print('train -- accuracy: %.3f ; f1 score: %.3f ;'
      'auroc: %.3f ; auprc: %.3f ; mse value: %.3f ;' %(final_acc_trn, final_f1_trn,
                                    final_auroc_trn, final_auprc_trn,
                                    final_mse_trn) )
print('test -- accuracy: %.3f ; f1 score: %.3f ;'
      'auroc: %.3f ; auprc: %.3f ; mse value: %.3f ;' %(final_acc_tst, final_f1_tst,
                                    final_auroc_tst, final_auprc_tst,
                                    final_mse_tst) )

################### Visualize #############################
# tree = rf.estimators_[5] # picking a random tree to visualise
# export_graphviz(tree, out_file = 'tree.dot', feature_names = labels, rounded = True, precision = 1)
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png(biomarker+'_'+task+'_tree.png')

# ## pruned tree
# rf_small = RandomForestClassifier(n_estimators=10, max_depth = 5)
# rf_small.fit(train_features, train_labels)# Extract the small tree
# tree_small = rf_small.estimators_[5]# Save the tree as a png image
# export_graphviz(tree_small, out_file = biomarker+'_'+task+'_cancDisc_small_tree.dot', feature_names = labels, rounded = True, precision = 1)
# (graph, ) = pydot.graph_from_dot_file(biomarker+'_'+task+'_cancDisc_small_tree.dot')
# graph.write_png(biomarker+'_'+task+'_cancDisc_small_tree.png');

################### Variable Importance ###################
# # Get numerical feature importances
# importances = list(rf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(labels, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances 
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# ## PLot
# # list of x locations for plotting
# x_values = list(range(len(importances)))
# # Make a bar chart
# plt.bar(x_values, importances, orientation = 'vertical', color='#520332')
# plt.xticks(x_values, labels, rotation=45, fontsize=4, ha='right')
# plt.ylabel('Importance'); plt.xlabel('Feature'); plt.title('Feature Importances');
# plt.savefig(task+'_VariableImportance.png')
