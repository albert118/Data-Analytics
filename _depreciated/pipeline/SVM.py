import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump

from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
################################################################################





################################################################################
def svc(df, *args, **kwargs):
	unique_test_name = 'StandardScaler SVC GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)

	y = df['QuoteConversion_Flag'].values
	IDs = df.Quote_ID
	X = df.drop(['QuoteConversion_Flag', 'Quote_ID'], axis=1).values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	MAX_ITER = np.arange(1000, 2000)
	_C = np.arange(1, 20)
	DUAL = ['True', 'False']
	TOL = 10.0 ** -np.arange(1, 8)
	N_COMPS = np.arange(2, 12)

	param_grid = [ {
		'svc__max_iter': MAX_ITER,
		'svc__C': _C,
		'svc__dual': DUAL,
		'svc__tol': TOL,
		'pca__n_components': N_COMPS,
		 },{
		'svc__max_iter': MAX_ITER,
		'svc__C': _C,
		'svc__dual': DUAL,
		'svc__tol': TOL,
		 } ]

	svc = svm.LinearSVC(random_state=42)
	pca = PCA()
	models = [make_pipeline(StandardScaler(), pca, svc, memory=memory), make_pipeline(StandardScaler(), svc)]

	grid = GridSearchCV(models, param_grid, cv=1000, iid=False, n_jobs=-1)

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