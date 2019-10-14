""" Demo the sampling methods in imblearn lib """
################################################################################
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


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
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split

################################################################################
df = pd.read_csv("preprocessing(1).csv")
ids = df['Quote_ID']
dates=df['Original_Quote_Date']

y = df.QuoteConversion_Flag.values
X = df.drop(['Unnamed: 0', 'Original_Quote_Date', 'QuoteConversion_Flag'],axis=1).values

# prints [class, count]
print(sorted(Counter(y).items()))

# smote_enn = SMOTEENN(random_state=0)
# X_resampled, y_resampled = smote_enn.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))

fields = ['Quote_ID', 'Field_info1', 'Field_info2', 'Field_info3', 'Field_info4', 'Coverage_info1', 'Coverage_info2', 'Coverage_info3', 
'Sales_info1', 'Sales_info2', 'Sales_info3', 'Sales_info4', 'Sales_info5', 'Personal_info1', 
'Personal_info2', 'Personal_info3', 'Personal_info4', 'Property_info1', 'Property_info3', 
'Property_info4', 'Property_info5', 'Geographic_info1', 'Geographic_info2', 'Geographic_info3', 
'Geographic_info4', 'Geographic_info5']

res=pd.DataFrame(data=X_resampled, columns=fields)
target=pd.DataFrame(data=y_resampled, columns=['QuoteConversion_Flag'])
res=pd.concat([target, res],axis=1)
res.Quote_ID=res.Quote_ID.astype("int64")
remain=pd.concat([ids, dates],axis=1)

fnl = pd.merge(res,remain,on='Quote_ID', left_index=True)

X = fnl.drop(['Original_Quote_Date', 'Quote_ID'],axis=1).values
target = fnl.QuoteConversion_Flag

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
print(principalDf.shape)
print(target.shape)

finalDf = pd.concat([principalDf, target], axis = 1)
var_ = sum(pca.explained_variance_ratio_)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_title('SMOTE with Tomek Links submethod', fontsize = 20)
colors = ['g', 'r']
targets = [0,1]

for target, color in zip(targets, colors):
	indicesToKeep = X['QuoteConversion_Flag'] == target
	ax.scatter(X.loc[indicesToKeep, 'principal component 1'], X.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

ax.legend(targets) 
ax.grid()
plt.show()

# fnl.to_csv("undersampled synthetic SMOTE with ENN (1).csv")