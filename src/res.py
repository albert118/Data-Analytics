# util and other methods
from prep import Prep
import pandas as pd
import numpy as np
from functools import wraps
import time
import os
from collections import Counter
import matplotlib.pyplot as plt

# sklearn
from sklearn.pipeline import (Pipeline, make_pipeline)
from sklearn.metrics import (confusion_matrix, f1_score, classification_report)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.metrics import roc_curve as ROC, auc

# Imblearn
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.ensemble import BalanceCascade

# Models
from sklearn.neural_network import MLPClassifier as MLP
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
# Three random forest methods 
from sklearn.ensemble import RandomForestClassifier as RAND
from sklearn.ensemble import ExtraTreesClassifier as XTRA
from sklearn.tree import DecisionTreeClassifier as DCIS

# Memory optimisation
from tempfile import mkdtemp
from shutil import rmtree
from joblib import (Memory, dump, load)

FIELDS = ['Quote_ID', 'Field_info1', 'Field_info2', 'Field_info3', 'Field_info4', 'Coverage_info1', 'Coverage_info2', 'Coverage_info3', 
	'Sales_info1', 'Sales_info2', 'Sales_info3', 'Sales_info4', 'Sales_info5', 'Personal_info1', 
	'Personal_info2', 'Personal_info3', 'Personal_info4', 'Property_info1', 'Property_info3', 
	'Property_info4', 'Property_info5', 'Geographic_info1', 'Geographic_info2', 'Geographic_info3', 
	'Geographic_info4', 'Geographic_info5']

RANDOM_STATE = None
CV = 10

################################################################################
# Utility methods
################################################################################
def timeit(method):
	@wraps(method)
	def wrap(*args, **kwargs):
		ts = time.time()
		result = method(*args, **kwargs)
		te =  time.time()
		print('function: {a} took {b:2.4f}s'.format(a=method.__name__.upper(), b=(te-ts)))
		return result
	return wrap

def save_model(model, results, grid_search, unique_test_name):
	if not os.path.exists(unique_test_name):
		os.system("mkdir %s" % unique_test_name)
		
	results.to_csv((unique_test_name+ '\\ida_a3_13611165.csv'), index=False)
	dump(model, (unique_test_name + '\\' + unique_test_name + '.joblib'))
	res = pd.DataFrame(grid_search.cv_results_)
	res.to_csv((unique_test_name + '\\grid_search_params.csv'), index=False)
	
	return

def print_results(unique_test_name, grid, y_pred, y_test):

	fpr, tpr, _ = ROC(y_test, y_pred)
	roc_auc = auc(fpr, tpr)
	
	print("-----------------%s--------------------" % unique_test_name)
	print("-----------------Best Param Overview--------------------")
	print("Best score: %0.4f" % grid.best_score_)
	print("Using the following parameters:")
	print(grid.best_params_)
	print("-----------------Scoring Model--------------------")
	print(classification_report(y_pred, y_test))
	print(confusion_matrix(y_pred, y_test), "\n")

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

	return

def tst_saving():
	X = pd.DataFrame([1,2,3,4,5,6], [1,2,3,4,5,6])
	y = pd.DataFrame([1,1,0,1,0,0])
	svc = svm.LinearSVC(random_state=42)
	svc.fit(X.values, y.values)
	y_pred = svc.predict(X.values)
	params = {'random_state':[2]}
	save_model(svc, pd.DataFrame(y_pred), GridSearchCV(svc, params, cv=2).fit(X.values, y.values), 'testing')
	return
################################################################################
# Meta methods
################################################################################
@timeit
def get_NN():
	df = Prep()
	X, y = smoter(df)
	_NN(X, y)
	return

@timeit
def get_SVM():
	df = Prep()
	X, y = smoter(df)
	_SVM(X, y)
	return

@timeit
def get_KNN():
	df = Prep()
	X, y = smoter(df)
	_KNN(X, y)
	return

@timeit
def get_TREES():
	df = Prep()
	X, y = smoter(df)
	_TREES(X, y)
	return

################################################################################
# Sampling methods
################################################################################
def smoter(df):
	IDs = df.Quote_ID
	target = df.QuoteConversion_Flag
	data = df.drop(['QuoteConversion_Flag'],axis=1).values
	print("Before SMOTE: ", sorted(Counter(target).items()))

	####
	# ENN
	####
	enn = ENN(sampling_strategy="not majority", kind_sel="mode", n_neighbors=5, n_jobs=-1, random_state=RANDOM_STATE)
	smote_enn = SMOTEENN(enn=enn, random_state=RANDOM_STATE)
	X_resampled, y_resampled = smote_enn.fit_resample(data, target)
	print("SMOTE ENN: ", sorted(Counter(y_resampled).items()))

	####
	# Tomeks
	####
	# smote_tomek = SMOTETomek(random_state=0)
	# X_resampled, y_resampled = smote_tomek.fit_resample(data, target)
	# print("Using SMOTE: ", sorted(Counter(y_resampled).items()))

	data = pd.DataFrame(data = X_resampled, columns = FIELDS)
	target = pd.DataFrame(data = y_resampled, columns = ['QuoteConversion_Flag'])

	return data, target

################################################################################
# Model methods
################################################################################
def _NN(X, y, *args, **kwargs):
	unique_test_name = 'MinMaxScaler MLP GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	IDs = X.Quote_ID
	X = X.drop(['Quote_ID'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	param_grid = {
		'mlp__activation': ["logistic", "relu"], 
		'mlp__alpha': 10.0 ** -np.arange(1, 8), 
		'mlp__hidden_layer_sizes': np.arange(5, 12), 
		'mlp__max_iter': np.arange(1000, 2001, 100),
	}

	mlp = MLP(random_state=1,solver='adam',learning_rate='adaptive',tol=1e-5)
	model_pipe = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), ('mlp', mlp)], memory=memory)
	grid = GridSearchCV(model_pipe, param_grid, cv=CV, iid=False, n_jobs=-1)

	grid.fit(X_train, np.ravel(y_train)) 
	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return

def _SVM(X, y, *args, **kwargs):
	unique_test_name = 'StandardScaler SVC GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	IDs = X.Quote_ID
	X = X.drop(['Quote_ID'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	MAX_ITER = [1200,1500,1800]
	_C = np.arange(1, 5)
	TOL = 10.0 ** -np.arange(1, 8)
	N_COMPS = np.arange(2, 10)

	param_grid_A = {'pca__n_components': N_COMPS, 'linearsvc__max_iter': MAX_ITER,'linearsvc__C': _C,'linearsvc__tol': TOL}
	param_grid_B = {'linearsvc__max_iter': MAX_ITER,'linearsvc__C': _C,'linearsvc__tol': TOL } 

	svc = svm.LinearSVC(random_state=RANDOM_STATE, fit_intercept=False)
	pca = PCA(iterated_power=25, random_state=RANDOM_STATE, whiten=True, svd_solver='full')
	model_pipe_A = make_pipeline(MinMaxScaler(), pca, svc)
	model_pipe_B = make_pipeline(MinMaxScaler(), svc)
	models = [model_pipe_A, model_pipe_B]
	params = [param_grid_A, param_grid_B]
	for i in range (0, len(models)):
		if i is 0:
			unique_test_name = unique_test_name+' with PCA'
		grid = GridSearchCV(models[i], params[i], cv=CV, iid=False, n_jobs=-1)
		grid.fit(X_train, np.ravel(y_train)) 
		prediction = grid.predict(X_test)
		print_results(unique_test_name, grid, prediction, y_test)

		results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])], axis=1)
		save_model(grid.best_estimator_, results, grid, unique_test_name)
	return

def _KNN(X, y, *args, **kwargs):
	unique_test_name = 'StandardScaler KNN GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	IDs = X.Quote_ID
	X = X.drop(['Quote_ID'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
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
	nca = NeighborhoodComponentsAnalysis(random_state=RANDOM_STATE, warm_start=False)
	knn = KNeighborsClassifier(n_jobs=-1)

	model = Pipeline(steps=[('scaler', StandardScaler()), ('nca', nca),('knn', knn)], memory=memory)
	grid = GridSearchCV(model, param_grid,cv=CV, iid=False, n_jobs=-1)
	grid.fit(X_train[:2200], np.ravel(y_train[:2200])) 

	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return

def _TREES(X, y, *args, **kwargs):
	
	# Create a temporary folder to store the transformers of the pipeline
	IDs = X.Quote_ID
	X = X.drop(['Quote_ID'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	randForest = RAND(n_jobs=-1, random_state=RANDOM_STATE)
	xtraForest = XTRA(n_jobs=-1, random_state=RANDOM_STATE)
	normForest = DCIS(random_state=RANDOM_STATE)
	
	param_grid_A = {'min_samples_split': np.arange(2, 8) }
	param_grid_B = {'max_depth': np.arange(2, 21),'min_samples_split': np.arange(1, 6),'n_estimators': np.arange(10, 110, 10) }
	param_grid_C = {'max_depth': np.arange(2, 21),'min_samples_split': np.arange(1, 6),'splitter': ['random', 'best'] }

	params = [param_grid_A, param_grid_B, param_grid_C]
	models = [randForest, xtraForest, normForest]
	names = ['randForest', 'xtraForest', 'decisionTree']

	for i in range(0, len(names)):
		unique_test_name = 'StandardScaler Tree %s GridSearchCV Optimised with SMOTE ENN' % names[i]
		
		cachedir = mkdtemp()
		memory = Memory(location=cachedir, verbose=10)

		model = make_pipeline(StandardScaler(), models[i], memory=memory)
		grid = GridSearchCV(model, params[i], cv=CV, iid=False, n_jobs=-1)
		grid.fit(X_train, np.ravel(y_train)) 
		
		prediction = grid.predict(X_test)
		print_results(unique_test_name, grid, prediction, y_test)
		results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])], axis=1)
		save_model(grid.best_estimator_, results, grid, unique_test_name)
		
		rmtree(mem)

	return