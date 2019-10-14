import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import (train_test_split, cross_val_score)
from joblib import dump
from sklearn.preprocessing import StandardScaler

# Three random forest methods 
from sklearn.ensemble import RandomForestClassifier as RAND
from sklearn.ensemble import ExtraTreesClassifier as XTRA
from sklearn.tree import DecisionTreeClassifier as DCIS

RAND_STATE = 42
################################################################################





################################################################################
def TREES(df, *args, **kwargs):
	y = df['QuoteConversion_Flag'].values
	IDs = df.Quote_ID
	X = df.drop(['QuoteConversion_Flag', 'Quote_ID'], axis=1).values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RAND_STATE)
	randForest = RAND(n_jobs=-1, max_depth=None, min_samples_split=2, random_state=RAND_STATE)
	xtraForest = XTRA(n_jobs=-1, n_estimators=10, max_depth=None, min_samples_split=2, random_state=RAND_STATE)
	normForest = DCIS(splitter="random", max_depth=None, min_samples_split=2, random_state=RAND_STATE)
	
	models = [make_pipeline(StandardScaler(), randForest), make_pipeline(StandardScaler(), xtraForest), make_pipeline(StandardScaler(), normForest)]
	
	for model in models:
		unique_test_name=''
		for key in model.named_steps:
			unique_test_name+= key + ' '

		print(unique_test_name)

		model.fit(X_train, y_train) 

		prediction = model.predict(X_test)
		print(classification_report(prediction, y_test))
		print(confusion_matrix(prediction, y_test), "\n")
		print("Score: ", model.score(X_test, y_test))

		prediction = pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])
		results = pd.concat([IDs, prediction], axis=1)

		fn = unique_test_name + "ida_a3_13611165.csv"
		results.to_csv(fn, index=False)
		dump(model, "FOREST[{}].joblib".format(unique_test_name))
	return