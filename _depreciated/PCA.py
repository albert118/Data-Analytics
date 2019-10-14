""" """
################################################################################
# Imports
import pandas as pd
import numpy as np
import mk3_analyser as mk3
from sklearn import preprocessing as pre
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
################################################################################
# 0.6834490398028756, single cluster per class
# ['Property_info5', 'Personal_info1', 'Sales_info1', 'Geographic_info4', 'Geographic_info1', 'Field_info4', 'Personal_info2', 'Sales_info4', 'Field_info3', 'Sales_info5', 'Sales_info3', 'Property_info4', 'Geographic_info5']

# # High clustering
# 0.7423198859140892
# ['Geographic_info2', 'Property_info4', 'Coverage_info1', 'Geographic_info3', 'Property_info3', 'Field_info4', 'Property_info3', 'Geographic_info3', 'Geographic_info1', 'Sales_info5', 'Geographic_info4', 'Field_info2', 'Property_info4', 'Personal_info4', 'Sales_info5', 'Field_info4', 'Sales_info2', 'Sales_info1', 'Field_info4', 'Geographic_info3', 'Sales_info5', 'Coverage_info3', 'Property_info3', 'Sales_info1', 'Sales_info5', 'Coverage_info2', 'Geographic_info5', 'Geographic_info4', 'Property_info5', 'Geographic_info1', 'Property_info3', 'Property_info5', 'Personal_info1', 'Property_info1', 'Property_info5']
################################################################################

# single clusters per target
# 0.8056940721994874
# ['Property_info1', 'Property_info5', 'Property_info1', 'Field_info1', 'Property_info5', 'Geographic_info2', 'Sales_info5', 'Geographic_info2', 'Property_info3', 'Field_info3', 'Geographic_info3', 'Field_info4', 'Geographic_info4', 'Personal_info4', 'Sales_info2', 'Geographic_info3', 'Geographic_info3', 'Property_info1', 'Geographic_info2', 'Property_info4', 'Personal_info4', 'Property_info4', 'Coverage_info3', 'Geographic_info2', 'Personal_info4', 'Coverage_info2', 'Personal_info3', 'Sales_info1', 'Sales_info5', 'Sales_info5', 'Sales_info3']
# remaining fields: lst=['Coverage_info1', 'Field_info2', 'Geographic_info1', 'Personal_info1', 'Personal_info2', 'Sales_info4']

# IDEA: find highest variance while reducing to two dimensions...
df = pd.read_csv("undersampled synthetic SMOTE with ENN (1).csv")
df = df.drop(['Unnamed: 0',  'Quote_ID', 'Original_Quote_Date'], axis=1)
target = pd.DataFrame(df.QuoteConversion_Flag)
tst = []
for field in df:
	tst.append(field)

for i in range(0, 100):
	for field in tst:
		drops=[]
		for i in range(0, 20):
			k = randint(0, len(tst)-1)
			if tst[k] not in drops:
				drops.append(tst[k])

		X = df.drop(drops, axis=1).values
		pca = PCA(n_components=2)

		principalComponents = pca.fit_transform(X)
		principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
		finalDf = pd.concat([principalDf, target], axis = 1)
		var_ = sum(pca.explained_variance_ratio_)

		if var_ < 0.75 and var_ >= 0.6:
			print(var_)
			print(drops)
			
			fig = plt.figure(figsize = (8,8))
			ax = fig.add_subplot(1,1,1)
			ax.set_ylabel('Principal Component 2', fontsize = 15)
			ax.set_xlabel('Principal Component 1', fontsize = 15)
			ax.set_title('2 component PCA', fontsize = 20)
			colors = ['r', 'g']
			targets = [0,1]
			for target, color in zip(targets,colors):
				indicesToKeep = finalDf['QuoteConversion_Flag'] == target
				ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

			ax.legend(targets) 
			ax.grid()
			del principalDf, finalDf
			plt.show()
			break

################################################################################


