import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from joblib import dump
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.preprocessing import StandardScaler

from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
################################################################################





################################################################################
def KNN(df, *args, **kwargs):
	unique_test_name = 'StandardScaler KNN GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)

	y = df['QuoteConversion_Flag'].values
	IDs = df.Quote_ID
	X = df.drop(['QuoteConversion_Flag', 'Quote_ID'], axis=1).values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	param_grid = {
			'knn__n_neighbours': np.arange(3, 12), 
			'knn__algorithm': ['ball_tree','kd_tree', 'brute'],
			'knn__leaf_size': np.arange(20, 30),
			'knn__p': [1, 2, 3, 4, 5],
			'nca__n_components': np.arange(2,12),
			'nca__max_iter': np.arange(1000, 2000),
			'nca__tol': 10.0 ** -np.arange(1, 8),
		}


	# model classes
	nca = NeighborhoodComponentsAnalysis(random_state=42, warm_start=False)
	knn = KNeighborsClassifier(n_jobs=-1)

	model = [make_pipeline(StandardScaler(), nca, knn, memory=memory)]
	
	grid = GridSearchCV(model, param_grid, cv=1000, iid=False, n_jobs=-1)

	grid.fit(X_train, y_train) 

	print("-----------------Best Param Overview--------------------")
	print("Best score: %0.4f" % grid.best_score_)
	print("Using the following parameters:")
	print(grid.best_params_)
	results = pd.DataFrame(grid.cv_results_)
	results.to_csv(unique_test_name+'_cv_results.csv', index=False)

	prediction = grid.predict(X_test)
	print("-----------------Scoring Model--------------------")
	print(classification_report(prediction, y_test))
	print(confusion_matrix(prediction, y_test), "\n")

	prediction = pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])
	results = pd.concat([IDs, prediction], axis=1)

	results.to_csv(unique_test_name + "ida_a3_13611165.csv", index=False)
	dump(grid, "MLP[{}].joblib".format(unique_test_name))
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return