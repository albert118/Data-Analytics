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


FIELDS = Prep().drop(columns=['Quote_Flag']).columns.tolist()  # the column names
RANDOM_STATE = None # probably controls seeding idk
CV = 10 # cross validation number, lower is quicker but higher is more robust

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
	fn = unique_test_name.replace(" ", "_")
	if not os.path.exists(fn):
		os.system("mkdir %s" % fn)
	
	results.to_csv(fn+ '\\ida_a3_12590941.csv', index=False)
	dump(model, (fn + '\\' + fn + '.joblib'))
	res = pd.DataFrame(grid_search.cv_results_)
	res.to_csv((fn + '\\grid_search_params.csv'), index=False)
	
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
	plt.plot(fpr, tpr,lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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

@timeit
def get_meta():
	df = Prep()
	X, y = smoter(df)
	models = ["modelA.joblib", "modelB.joblib", "modelC.joblib", "modelD.joblib"]
	model_list=[]
	for model in models:
		model_list.append(load(model))
	_meta_pred(X, y, model_list)
	return

def gen_Preds(model):
	df = Prep('test')
	IDs = df.Quote_Id
	X = df.drop(['Quote_Id'], axis=1).values
	prediction = model.predict(X)
	results = pd.DataFrame(data=prediction, columns=['Quote_Flag'])
	results = pd.concat([IDs, results], axis=1)
	results.to_csv("ida_a3_12590941.csv", index=False)
	return

def gen_meta_preds():
	df = Prep('test')
	IDs = df.Quote_Id
	X = df.drop(['Quote_Id'], axis=1).values

	models = ["modelA.joblib", "modelB.joblib", "modelC.joblib", "modelD.joblib"]
	model_list=[]
	
	for model in models:
		model_list.append(load(model))

	meta_res = pd.DataFrame(IDs)
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	for model in model_list:
		meta_res = pd.concat([meta_res, pd.DataFrame(model.predict(X))], axis=1)

	meta = load("meta.joblib")
	prediction = meta.predict(meta_res.drop(["Quote_Id"],axis=1))
	results = pd.DataFrame(data=prediction, columns=['Quote_Flag'])
	results = pd.concat([IDs, results], axis=1)
	results.to_csv("ida_a3_12590941.csv", index=False)
	return

################################################################################
# Sampling methods
################################################################################
def smoter(df):
	IDs = df.Quote_Id
	target = df.Quote_Flag

	data = df.drop(['Quote_Flag'],axis=1).values
	print("Before SMOTE: ", sorted(Counter(target).items()))

	enn = ENN(sampling_strategy="not majority", kind_sel="mode", n_neighbors=5, n_jobs=-1)
	smote_enn = SMOTEENN(enn=enn, random_state=RANDOM_STATE)
	X_resampled, y_resampled = smote_enn.fit_resample(data, target)
	print("SMOTE ENN: ", sorted(Counter(y_resampled).items()))

	data = pd.DataFrame(data = X_resampled, columns = FIELDS)
	target = pd.DataFrame(data = y_resampled, columns = ['Quote_Flag'])

	return data, target

################################################################################
# Model methods
################################################################################
def _NN(X, y, *args, **kwargs):
	unique_test_name = 'MinMaxScaler MLP GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	IDs = X.Quote_Id
	X = X.drop(['Quote_Id'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	param_grid = {
		'mlp__activation': ["logistic"], 
		'mlp__alpha': [0.1, 0.01],
		'mlp__hidden_layer_sizes': [(10, 2), (10, 4)],
		'mlp__max_iter': [500],
		'mlp__beta_1' : [0.8, 0.9],
		'mlp__beta_2': [0.88, 0.90, 0.99],
	}

	mlp = MLP(early_stopping=True, validation_fraction = 0.25, random_state=RANDOM_STATE, solver='adam',learning_rate='adaptive')
	model_pipe = Pipeline(steps=[('StandardScaler', StandardScaler()), ('mlp', mlp)], memory=memory)
	grid = GridSearchCV(model_pipe, param_grid, cv=CV, n_jobs=-1)

	grid.fit(X_train, np.ravel(y_train)) 
	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['Quote_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return

def _SVM(X, y, *args, **kwargs):
	unique_test_name = 'StandardScaler SVC GridSearchCV Optimised with SMOTE ENN'
	IDs = X.Quote_Id
	X = X.drop(['Quote_Id'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	MAX_ITER = [1800]
	_C = np.arange(1, 5)
	TOL = 10.0 ** -np.arange(1, 6)
	N_COMPS = np.arange(2, 8)

	param_grid_A = {'pca__n_components': N_COMPS, 'linearsvc__max_iter': MAX_ITER,'linearsvc__C': _C,'linearsvc__tol': TOL}
	param_grid_B = {'linearsvc__max_iter': MAX_ITER,'linearsvc__C': _C,'linearsvc__tol': TOL } 

	svc = svm.LinearSVC(random_state=RANDOM_STATE, fit_intercept=False)
	pca = PCA(iterated_power=25, random_state=RANDOM_STATE, whiten=True, svd_solver='full')
	

	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)

	unique_test_name = unique_test_name+' with PCA'
	model_pipe_A = make_pipeline(MinMaxScaler(), pca, svc, memory=memory)
	
	grid = GridSearchCV(model_pipe_A, param_grid_A, cv=CV, n_jobs=-1)
	grid.fit(X_train, np.ravel(y_train)) 
	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['Quote_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	rmtree(cachedir)

	###############
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	unique_test_name = 'StandardScaler SVC GridSearchCV Optimised with SMOTE ENN'

	model_pipe_B = make_pipeline(MinMaxScaler(), svc, memory=memory)
	
	grid = GridSearchCV(model_pipe_B, param_grid_B, cv=CV, n_jobs=-1)
	grid.fit(X_train, np.ravel(y_train)) 
	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['Quote_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	rmtree(cachedir)

	return

def _KNN(X, y, *args, **kwargs):
	unique_test_name = "StandardScaler KNN GridSearchCV Optimised with SMOTE ENN"
	if "unique_test_name" in kwargs:
		unique_test_name += kwargs["unique_test_name"]
	
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	IDs = X.Quote_Id
	X = X.drop(['Quote_Id'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3000, train_size=3000, random_state=RANDOM_STATE)
	
	param_grid = {
			'knn__n_neighbors': np.arange(3, 11), 
			'knn__algorithm': ['ball_tree','kd_tree'],
			'knn__leaf_size': np.arange(10, 25),
			'nca__n_components': np.arange(2, X.shape[1]),
			'nca__max_iter': [1200, 1500, 1800, 2000],
			'nca__tol': 10.0 ** -np.arange(1, 6)
		}

	# model classes
	nca = NeighborhoodComponentsAnalysis(random_state=RANDOM_STATE, warm_start=False)
	knn = KNeighborsClassifier(n_jobs=-1)

	model = Pipeline(steps=[('scaler', StandardScaler()), ('nca', nca),('knn', knn)], memory=memory)
	grid = GridSearchCV(model, param_grid,cv=CV, n_jobs=-1)
	grid.fit(X_train[:2200], np.ravel(y_train[:2200])) 

	prediction = grid.predict(X_test)
	print_results(unique_test_name, grid, prediction, y_test)

	results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['Quote_Flag'])], axis=1)
	save_model(grid.best_estimator_, results, grid, unique_test_name)
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return

def _TREES(X, y, *args, **kwargs):
	
	# Create a temporary folder to store the transformers of the pipeline
	IDs = X.Quote_Id
	X = X.drop(['Quote_Id'], axis=1).values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)
	
	randForest = RAND(n_jobs=-1, random_state=RANDOM_STATE)
	xtraForest = XTRA(n_jobs=-1, random_state=RANDOM_STATE)
	normForest = DCIS(random_state=RANDOM_STATE)

	param_grid_A = {'randomforestclassifier__min_samples_split': np.arange(0.005, 0.02), 'randomforestclassifier__n_estimators': np.arange(90, 160, 10)}
	param_grid_B = {'extratreesclassifier__max_depth': np.arange(2, 21),'extratreesclassifier__min_samples_split': np.arange(0.005, 0.02), 'extratreesclassifier__n_estimators': np.arange(10, 110, 10) }
	param_grid_C = {'decisiontreeclassifier__max_depth': np.arange(2, 21),'decisiontreeclassifier__min_samples_split': np.arange(0.005, 0.02), 'decisiontreeclassifier__splitter': ['random', 'best'] }

	params = [param_grid_A, param_grid_B, param_grid_C]
	models = [randForest, xtraForest, normForest]
	names = ['randForest', 'xtraForest', 'decisionTree']

	for i in range(0, len(names)):
		unique_test_name = 'StandardScaler Tree %s GridSearchCV Optimised with SMOTE ENN' % names[i]
		if "unique_test_name" in kwargs:
			unique_test_name += kwargs["unique_test_name"]

		cachedir = mkdtemp()
		memory = Memory(location=cachedir, verbose=10)

		model = make_pipeline(StandardScaler(), models[i], memory=memory)
		grid = GridSearchCV(model, params[i], cv=CV, n_jobs=-1)
		grid.fit(X_train, np.ravel(y_train)) 
		
		prediction = grid.predict(X_test)
		print_results(unique_test_name, grid, prediction, y_test)
		results = pd.concat([IDs, pd.DataFrame(data=prediction, columns=['Quote_Flag'])], axis=1)
		save_model(grid.best_estimator_, results, grid, unique_test_name)
		
		rmtree(memory)

	return

def _meta_pred(X, y, model_list, *args, **kwargs):
	unique_test_name = " Meta classifier"
	# create blank data frame with IDs then merge all classifier predictions into one dataframe
	IDs = X.Quote_Id
	X_test = X.drop(["Quote_Id"], axis=1)
	meta_res = pd.DataFrame(IDs)

	for model in model_list:
		meta_res = pd.concat([meta_res, pd.DataFrame(model.predict(X_test))], axis=1)

	_KNN(meta_res, y, unique_test_name=unique_test_name)
	_TREES(meta_res, y, unique_test_name=unique_test_name)

	return
