# Models that must be used "tree", 'KNN', 'Random Forest', 'SVM', 'ANN'

# SVM, done!! Performs 72.104% without PCA and ENN SMOTE

# preprocess
from preprocessing import prep
# now call our fitting model(s)
from SVM import svc
from KNN import KNN
from MLP import NN
from FOREST import TREES
from functools import wraps
import time

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

@timeit
def run():
	df_pre = prep()
	#df_B = svc(df_pre)
	#df_C = KNN(df_pre)
	df_D =  NN(df_pre)
	#df_E = TREES(df_pre)
 