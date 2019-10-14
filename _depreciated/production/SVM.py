import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import load
################################################################################





################################################################################
def svc(df, *args, **kwargs):
	# Prep input data for model use (2D array of data)
	# data = df_new.as_matrix(columns=df_new.columns[2:])
	# for feature in data:
	# 	feature=feature.transpose()

	# Now get the data vals of f(x)=y, where y is our QuoteConversionFlag and X is remaining fields
	IDs = df.Quote_ID
	X = df.drop(['Quote_ID'], axis=1).values
	
	models = []
	for model in ['SVM[standardscaler linearsvc ].joblib', 'SVM[standardscaler pca linearsvc ].joblib']:
		models.append(load(model))
		

	for model in models:
		# model = svm.SVC(kernel="linear")
		unique_test_name=''
		for key in model.named_steps:
			unique_test_name+= key + ' '

		print(unique_test_name)

		# set_params(svc__C=.1, svc__gamma=0, svc__max_iter=100)
		prediction = model.predict(X)
		prediction = pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])
		results = pd.concat([IDs, prediction], axis=1)
		fn = unique_test_name + "ida_a3_13611165.csv"
		results.to_csv(fn, index=False)
	return results