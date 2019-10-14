import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.preprocessing import StandardScaler
################################################################################





################################################################################
def KNN(df, *args, **kwargs):
	IDs = df.Quote_ID
	X = df.drop(['Quote_ID'], axis=1).values
	
	models = []
	for model in ['KNN[standardscaler kneighborsclassifier ].joblib', 'KNN[standardscaler neighborhoodcomponentsanalysis kneighborsclassifier ].joblib']:
		models.append(load(model))
		
	for model in models:
		unique_test_name=''
		for key in model.named_steps:
			unique_test_name+= key + ' '

		print(unique_test_name)
		prediction = pd.DataFrame(data=model.predict(X), columns=['QuoteConversion_Flag'])
		results = pd.concat([IDs, prediction], axis=1)
		fn = unique_test_name + "ida_a3_13611165.csv"
		results.to_csv(fn, index=False)
	return