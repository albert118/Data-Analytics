import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def fit_PCA(data):
	# get our components of prepped data
	targets = data.QuoteConversion_Flag
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data)
	# cast to df
	principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
	# concat the target field
	finalDf = pd.concat([principalDf, targets], axis = 1)
	var_ = sum(pca.explained_variance_ratio_)
	print("Variance of compression: ", var_)
	return var_, finalDf

def reduction(df, *args, **kwargs):
	# read preprocessed data
	if 'file' in args:
		df = pd.read_csv("sampled(2).csv")
	# drop unnecessary rows, these shouldn't be compressed!!!

	# get user to input variables to drop before PCA..
	drops=['Geographic_info2', 'Geographic_info1', 'Sales_info4', 'Coverage_info3', 
	'Property_info1', 'Property_info3', 'Personal_info1', 'Coverage_info1', 'Property_info5', 
	'Personal_info4', 'Sales_info2', 'Sales_info5', 'Geographic_info4', 'Personal_info2']

	IDs = df.Quote_ID
	readd = df[drops]
	df = df.drop(['Quote_ID'], axis=1)
	df = df.drop(drops, axis=1)

	var_, df = fit_PCA(df)
	df = pd.concat([IDs, readd, df], axis=1)
	# save with ID's is easier when uploading to Kaggle later
	if 'file' in args:
		df.to_csv("comp(3).csv", index=False)
	return df
