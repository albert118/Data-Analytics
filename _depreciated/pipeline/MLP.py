import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix,f1_score,classification_report)
from sklearn.model_selection import (train_test_split, GridSearchCV)
from joblib import dump
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.neural_network import MLPClassifier as MLP

from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
################################################################################

# Best score: 0.8784
# Using the following parameters:
# {'mlp__activation': 'logistic', 'mlp__alpha': 1e-06, 'mlp__hidden_layer_sizes': 11, 'mlp__max_iter': 1500}
# -----------------Scoring Model--------------------
#               precision    recall  f1-score   support

#            0       0.89      0.89      0.89     20999
#            1       0.87      0.87      0.87     18379

#     accuracy                           0.88     39378
#    macro avg       0.88      0.88      0.88     39378
# weighted avg       0.88      0.88      0.88     39378

# [[18638  2361]
#  [ 2356 16023]]

# function: RUN took 18564.0860s



################################################################################
def NN(df, *args, **kwargs):

	unique_test_name = 'MinMaxScaler MLP GridSearchCV Optimised with SMOTE ENN'
	# Create a temporary folder to store the transformers of the pipeline
	cachedir = mkdtemp()
	memory = Memory(location=cachedir, verbose=10)
	
	y = df['QuoteConversion_Flag'].values
	IDs = df.Quote_ID
	X = df.drop(['QuoteConversion_Flag', 'Quote_ID'], axis=1).values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
	param_grid = {
		'mlp__activation': ["logistic", "relu"], 
		'mlp__alpha': 10.0 ** -np.arange(1, 8), 
		'mlp__hidden_layer_sizes': np.arange(5, 12), 
		'mlp__max_iter': [500,1000,1500],
	}

	mlp = MLP(random_state=1,solver='adam',learning_rate='adaptive',tol=1e-5)
	model_pipe = Pipeline(steps=[('minmax_scaler', MinMaxScaler()), ('mlp', mlp)], memory=memory)
	grid = GridSearchCV(model_pipe, param_grid, cv=10, iid=False, n_jobs=-1)

	print(unique_test_name)

	grid.fit(X_train, y_train) 

	print("-----------------Best Param Overview--------------------")
	print("Best score: %0.4f" % grid.best_score_)
	print("Using the following parameters:")
	print(grid.best_params_)

	prediction = grid.predict(X_test)
	print("-----------------Scoring Model--------------------")
	print(classification_report(prediction, y_test))
	print(confusion_matrix(prediction, y_test), "\n")

	prediction = pd.DataFrame(data=prediction, columns=['QuoteConversion_Flag'])
	results = pd.concat([IDs, prediction], axis=1)

	fn = unique_test_name + "ida_a3_13611165.csv"
	results.to_csv(fn, index=False)
	dump(grid, "MLP[{}].joblib".format(unique_test_name))
	
	# Delete the temporary cache before exiting
	rmtree(cachedir)
	return