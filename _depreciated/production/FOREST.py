import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import (train_test_split, cross_val_score)
from joblib import load
from sklearn.preprocessing import StandardScaler

# Three random forest methods 
from sklearn.ensemble import RandomForestClassifier as RAND
from sklearn.ensemble import ExtraTreesClassifier as XTRA
from sklearn.tree import DecisionTreeClassifier as DCIS

RAND_STATE = 42
################################################################################





################################################################################
def TREES(df, *args, **kwargs):
	IDs = df.Quote_ID
	X = df.drop(['Quote_ID'], axis=1).values

	
	models = []
	for model in ['FOREST[standardscaler decisiontreeclassifier ].joblib.', 'FOREST[standardscaler extratreesclassifier ].joblib','FOREST[standardscaler randomforestclassifier ].joblib',]:
		models.append(load(model))

	for model in models:
		unique_test_name=''
		for key in model.named_steps:
			unique_test_name+= key + ' '

		print(unique_test_name)
		prediction = model.predict(X)
		prediction = pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])
		results = pd.concat([IDs, prediction], axis=1)
		fn = unique_test_name + "ida_a3_13611165.csv"
		results.to_csv(fn, index=False)
	return