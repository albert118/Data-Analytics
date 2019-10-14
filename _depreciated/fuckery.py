import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split

####
# Fuckery for predictions of PCA 
####

####
# This looks sick
# 0.906468904600014
# drops=['Personal_info1', 'Sales_info3', 'Sales_info5', 'Property_info5', 'Geographic_info2', 'Coverage_info2', 'Field_info4', 'Sales_info1', 'Property_info1', 'Sales_info4', 'Geographic_info5', 'Sales_info2', 'Geographic_info3', 'QuoteConversion_Flag', 'Personal_info2', 'Property_info4', 'Property_info3']
####


####
# but this actually segregates
# 0.6245340831616986

def fit_PCA(data):
	# get our components of prepped data
	targets = df.QuoteConversion_Flag
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data)
	# cast to df
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
	# concat the target field
	finalDf = pd.concat([principalDf, targets], axis = 1)
	var_ = sum(pca.explained_variance_ratio_)
	print("Variance of compression: ", var_)
	return var_, finalDf

df = pd.read_csv("sampled(2).csv")

drops=['Geographic_info2', 'Geographic_info1', 'Sales_info4', 'Coverage_info3', 
'Property_info1', 'Property_info3', 'Personal_info1', 'Coverage_info1', 'Property_info5', 
'Personal_info4', 'Sales_info2', 'Sales_info5', 'Geographic_info4', 'Personal_info2']

targets = df.QuoteConversion_Flag
df = df.drop(['Unnamed: 0', 'Quote_ID', 'Original_Quote_Date'], axis=1)
df = df.drop(drops, axis=1)

var_, df_new = fit_PCA(df)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_title('SMOTE with Tomek Links submethod', fontsize = 20)
colors = ['g', 'r']
classes = [0, 1]

for target, color in zip(classes, colors):
	indicesToKeep = df_new['QuoteConversion_Flag'] == target
	ax.scatter(df_new.loc[indicesToKeep, 'principal component 1'], df_new.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

ax.legend(classes)
ax.grid()
plt.show()

# before class balancing this should achieve 80% accuracy
#             precision    recall  f1-score   support

#           0       0.81      1.00      0.89     63315
#           1       0.00      0.00      0.00     14910

# avg / total       0.66      0.81      0.72     78225

# Using SMOTE
#               precision    recall  f1-score   support

#            0       0.75      0.30      0.43     15597
#            1       0.78      0.96      0.86     41366

#     accuracy                           0.78     56963
#    macro avg       0.77      0.63      0.65     56963
# weighted avg       0.78      0.78      0.75     56963

# Confusion matrix with SMOTE ENN
# metrics.confusion_matrix(target, prediction)
# array([[ 4677, 10920],
#        [ 1540, 39826]], dtype=int64)

# Normalised...
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# >>> cm
# array([[0.29986536, 0.70013464],
#        [0.03722864, 0.96277136]])

# Using SMOTE ENN, test splitting followed by PCA yields:
#               precision    recall  f1-score   support

#            0       0.28      0.11      0.16      5218
#            1       0.72      0.89      0.80     13580

#     accuracy                           0.67     18798
#    macro avg       0.50      0.50      0.48     18798
# weighted avg       0.60      0.67      0.62     18798