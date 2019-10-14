# Models that must be used "tree", 'KNN', 'Random Forest', 'SVM', 'ANN'

# SVM, done!! Performs 72.104% without PCA and ENN SMOTE

# preprocess
from preprocessing import prep
# sample with ensemble and fix class distribution
from sampler import sampler
# reduce components via PCA
from PCA import reduction
# quickly graph our results from PCA
import quick_grapher
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
	df_sampled = sampler(df_pre)
	# df_reduced_dims = reduction(df_sampled)
	# quick_grapher.graph(df_reduced_dims)

	# Model fitting with SVM
	# df_B = svc(df_sampled)
	# df_C = KNN(df_sampled)
	df_D =  NN(df_sampled)
	#df_E = TREES(df_sampled)
 