""" Prototype analyser : MK2
IDEA 	: take data "objects" and generate overview info on them graphically.
mk3		: automate visualisation process for analytics

Author : Albert Ferguson """

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from functools import wraps

test_str_format = "################################################################################\n{}\n################################################################################\n"

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

################################################################################
# Data retrieval and writing methods
################################################################################

def read_xlsx(file_name, sheet, *args, **kwargs):
	""" 
	Get XLSX sheet data and return it as a pandas dataframe for further use

	kwargs:
	sheet_name:	the sheet name to read data from into the frame.
	"""
	return(pd.read_excel(file_name, sheet_name=sheet))
	 
def write_xlsx(outs, *args, **kwargs):
	""" 
	Create a new pandas data frames from input data and write to new sheet
	in existing XLSX file, or default to create a new one to avoid 
	overwrite issue.

	Note: Throws IOError on file_name kwarg errors.
	
	args:
	-O : 		over ride arg, will attempt to write data to existing XLSX

	kwargs:
	file_name: 	if -O is included, a file name is expected to attach
				the new data sheet.
	"""

	df = pd.DataFrame(outs)

	try:
		if "-O" in args:
				df.to_excel(kwargs.get("file_name"), sheet_name="output dataframe", engine="xlsxwriter")
		else:
			df.to_excel("output.xlsx", engine="xlsxwriter")
	except IOError:
			if "file_name" in kwargs:
				print("File name not found or invalid.\nPlease check the file name")
			else:
				print("The keyword arg [file_name] was not specified...\nPlease specify to continue.")
	return

################################################################################
# Weighting and binning methods
################################################################################

def EWMA(inputs, *args, **kwargs):
	"""
	Exponentially Weighted Moving Average (EWMA)  of an inputted list, returns 
	the weighted list. 

	- kwargs:
	alpha : defines the alpha weight-update constant. Must be on range [0,1].
			default: 0.5

	returns:
	a numpy data array of data type int16
	"""

	if "alpha" in kwargs:
		alpha = kwargs.get("alpha")
		if alpha > 1 or alpha < 0:
			print("Alpha value cannot exced range of [0,1]!!!")
			return
	else:
		alpha = 0.5 # init, default alpha

	S1 = inputs.sum()/len(inputs)
	weighted_list = [S1]

	for i in range(1, len(inputs)):
		St = alpha * inputs[i] + (1-alpha) * weighted_list[i-1]
		weighted_list.append(St)

	return np.array(weighted_list, dtype=np.int32)

def equiwidth(data, k, *args, **kwargs):
	"""
	Bin the data into k equally sized bins. This algorithm follows the widely accepted 
	equal-width binning method.
	The following series: min+w, min2w, ... , min+(k-1)w, min+kw is followed to determine the
	interval boundries.

	This algorithm supports input data from either Python's default list type or Numpy's array type.
	----------------------------------------------------------------------------------
	Note: these boundaries are inclusive non inclusive, i.e. [min+w, min+2w[ or
	[min+w, min+2w) - in other symbols.
	----------------------------------------------------------------------------------

	- data:
	 	is an inputted list of data to be binned.
	- k:
		is the integer value equal to the number of desired bins.

	- args:
		-D:
			debug flag. Print debug statements to console.

	- kwargs:
		- kind = {'quicksort', 'mergesort', 'heapsort', 'stable'}, default is 'stable'.
		This is a passthrough kwarg for Numpy's numpy.sort() function, see below:
		----------------------------------------------------------------------------------
			optional Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
	        and 'mergesort' use timsort or radix sort under the covers and, in general,
	        the actual implementation will vary with data type. The 'mergesort' option
	        is retained for backwards compatibility.
	    ----------------------------------------------------------------------------------

	returns:
		- 	list of bin edges, including left edge of first bin and right edge of last bin.  
        	All but the last (righthand-most) bin is half-open.
	"""
	
	data_range = abs(data.min()-data.max())
	w = math.floor(data_range/(k+1))
	g = w
	i = 0

	if "-D" in args:
		print("width:{}\trange:{}".format(w, data_range)) 

	if(type(data) == type(np.array([1]))):
		if "kind" in kwargs:
			sort_input = np.sort(data, kind=kwargs.get("kind"))
		else:
			sort_input = np.sort(data, kind='stable')

	
	else:
		sort_input = data.sort()

	dp = sort_input[i]
	binned, bins, edges = [], [], []
	edges.append(data.min())

	if "-D" in args:
		print("Utilising {} type.\nFirst data point: {}\nInitialised bins.\n".format(type(sort_input), dp))

	while dp < data.min()+g:
		binned.append(dp)
		i+=1
		dp = sort_input[i]
		
		if dp < data.min()+g:
			pass
		else:
			bins.append(np.array(binned).astype(float))
			binned = []
			g+=w
			edges.append(data.min()+g)
		if g == (k+1)*w:
			break

	if "-D" in args:
		print("Total Bins Created: {}\nDesired Bins: {}\n".format(len(bins), k))
		print("Final boundry value: {}\n".format(sort_input.min()+g))
		print("Returning bins...")

	return edges

def equidepth(data, k, *args, **kwargs):
	"""
	Bin the data into k equally deep bins. This algorithm uses a basic implementation
	of the "intuitive" logic used in visual interpretation of bin counts with histograms.

	This algorithm determines the length of the set and takes the floor value
	of the length over k to determine the integer count of each bin. Values are then 
	added to each bin from a sorted input (param: data) list.

	Excess values (from possible rounding differences) are added one at a time to each bin
	in order of bin creation until none remain.

	- data:
	 	is an inputted list of data to be binned.
	- k:
		is the integer value equal to the number of desired bins.

	- args:
		-D:
			debug flag. Print debug statements to console.

	- kwargs:
		- kind = {'quicksort', 'mergesort', 'heapsort', 'stable'}, default is 'stable'.
		This is a passthrough kwarg for Numpy's numpy.sort() function, see below:
		----------------------------------------------------------------------------------
			optional Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
	        and 'mergesort' use timsort or radix sort under the covers and, in general,
	        the actual implementation will vary with data type. The 'mergesort' option
	        is retained for backwards compatibility.
	    ----------------------------------------------------------------------------------

	returns:
		a nested list of binned elements where index is equivalent to bin number.
	"""

	bin_length = math.floor(len(data)/(k+1))
	bins, binned = [], []
	bin_count = 0
	max_binned_idx = 0
	
	if(type(data) == type(np.array([1]))):
		if "kind" in kwargs:
			sort_input = np.sort(data, kind=kwargs.get("kind"))
		else:
			sort_input = np.sort(data, kind='stable')
	else:
		sort_input = data.sort()

	if "-D" in args:
		print("Utilising {} type.\nFirst data point: {}\nInitialised bins.\n".format(type(sort_input), sort_input[0]))

	for dp in sort_input:
		if bin_count < bin_length:
			binned.append(dp)
			bin_count+=1
			max_binned_idx+=1
		elif len(bins) < (k+1):
			bins.append(np.array(binned).astype(float))
			binned = []
			bin_count = 0
			binned.append(dp)
			max_binned_idx+=1
		else:
			break

	if max_binned_idx != len(data)-1:
		bin_dist_count = 0
		for i in range(max_binned_idx+1, len(data)):
			if i < len(bins):
				bins[i].append(sort_input[i])
				max_binned_idx+=1

	if "-D" in args:
		print("Total Bins Created: {}\nDesired Bins: {}\n".format(len(bins), k))
		print("Returning bins...")

	return bins

################################################################################
# Visualiser Functions
################################################################################
def histogram(data, mu, sigma, *args, **kwargs):
	"""
	Generate a histogram using specified formating consistencies.
	Default figure output file anme is frequency_analysis.png, if the input_name
	keyword arg is defined, then the file name is set 
	to input_name + "frequency_analysis.png"

	args:
	-F:			frequency distribution, if true a frequeny distribution 
				is created. Else, a probability distribution is created.

	-D:			Generate max verbosity debug output.

	kwargs:
	input_name:	the input data str name. This is passed explicitly as __str__
				of most data passed (lists, dataframes, dicts...) would be 
				innacurate/indescriptive for titles.

	hist_title:	title for the histogram generated.

	returns tuple:

	save_title:	file name that output figure is written to.
	figure	  : current figure and details

	"""

	plt.figure()

	if "bins" in kwargs:
		bins = kwargs.get("bins")
	else:
		bins = False

	if "-F" in args and bins == True:
		binned_input, binned_ranges, patches = plt.hist(data, bins=bins, density=False,facecolor='gray')
		plt.ylabel("Frequency")
	elif bins is not False and '-F' in args:
		binned_input, binned_ranges, patches = plt.hist(data, bins=bins, density=False,facecolor='gray')
		plt.ylabel("Frequency")
	else:
		binned_input, binned_ranges, patches = plt.hist(data, density=True,facecolor='gray')
		plt.ylabel("Probability")

	if "input_name" in kwargs:
		input_name = kwargs.get("input_name")
		hist_title = input_name + " Frequency Analysis"
		save_title = input_name + " frequency_analysis.png"
		plt.xlabel(input_name)
	else:
		hist_title = "Frequency Analysis"
		save_title = " frequency_analysis.png"
		plt.xlabel("Input Data")

	if "hist_title" in kwargs:
		plt.title(kwargs.get("hist_title"))

	if "-D" in args:
		print("Binned input:\n", binned_input)
		print("Patches:\n", patches)
		print("Ranges:\n", binned_ranges)
	else:
		pass

	x_posit = binned_ranges[1]
	y_posit = binned_input.max()
	plt.text(x_posit, y_posit, fr"$\mu={mu},\ \sigma={sigma}$")
	plt.grid(True)
	# set relative to dataset input range
	plt.xlim(data.min(), data.max())
	plt.savefig(save_title)

	return save_title, plt.figure

def scatter():
	pass
def scatter_matrix():
	pass
################################################################################
# Auto-visualiser Factories
################################################################################

def auto_run_hists(data, data_name, test_range, *args, **kwargs):
	"""
	Implement the histogram and binning methods defined by the mk3_analyser
	automatically across a range of possible bins.
	----------------------------------------------------------------------------
	auto_run_hists iterates through possible binnings of the data argument across 
	a user defined range and provides consistent formatting of the data as defined 
	by the histogram() function. 
	Further, each graph includes the data's mean and standard deviation 
	properties for convenient overview alongside the graphical interpretation.

	parameters:
	-data:		The data set to be supplied. This is a single dimensionsal list, 
				series or array. It is expected that the data is orignated from 
				the read_xlsx function which returns a pandas dataframe - of 
				which the columns are pandas.Series class structures. 
				These are converted to numpy arrays.

	data_name:	The input data str name. This is passed explicitly as __str__
				of most data passed (lists, dataframes, dicts...) would be 
				innacurate/indescriptive for titles.

	test_range:	The range of possible bin values. This defines the discrete 
				integer values that bins are calculated for the data set 
				and visualised.

	----------------------------------------------------------------------------
	args for the auto function are passed to the histogram when it is called. 
	Therefore, all possible arg values of histogram are implemented by auto_run_hists.
	----------------------------------------------------------------------------
	args:
	-F:			frequency distribution, if true a frequeny distribution 
				is created. Else, a probability distribution is created.

	-D:			Generate max verbosity debug output.

	kwargs:
				bin_method = {'equiwidth', 'equidepth'} this defines the 
				choice of binning operation performed. 
				Default is equiwidth.
	"""

	mu = sum(data)/len(data)
	sigma = np.std(data)

	if type(data) == type(pd.Series()):
		data = data.to_numpy()

	for i in range(test_range[0], test_range[1]):
		title = data_name + " - iteration - " + str(i)

		if "equidepth" in kwargs:
			edges = equidepth(data, i)
			if '-D' in args:
				print(edges)
		else:
			edges = equiwidth(data, i)

		print("\tBins Generated: ", i)
		save_fn, fig = histogram(data, mu, sigma, *args, bins=edges, hist_title=title, input_name=(data_name + ', ' + str(i)))
		print("saved histogram to: ", save_fn)
		plt.show()
		edges = []
	return

################################################################################
# Tests
################################################################################
@timeit
def test_all():	
	test_EWMA()
	test_hist()
	print(test_str_format.format("\t\t\tBinning Tests"))
	test_equiwidth()
	test_equidepth()
	print(test.format("\t\t\tVisualiastion Factory Tests"))
	test_auto_hists()
	return True

@timeit
def test_EWMA():
	print(test_str_format.format("\t\t\tEWMA test"))
	mu, sigma = 100, 15
	x = mu + sigma * np.random.randn(10000)
	try:
		x = EWMA(x)
		print(x[0:10])
		return True
	except Exception as e:
		print(e)
		return False

@timeit
def test_hist():
	print(test_str_format.format("\t\t\tHistogram test"))
	mu, sigma = 100, 15
	x = mu + sigma * np.random.randn(10000)
	try:
		histogram(x, mu, sigma,'-F', '-D')
		plt.show()
		return True
	except Exception as e:
		print(e)
		return False

@timeit
def test_equiwidth():
	mu, sigma = 100, 15
	x = mu + sigma * np.random.randn(10000)
	print("\n** Equi-Width Binning **\n")
	try:
		bin_num = 10
		tst = equiwidth(x, bin_num, '-D')
		print(tst[0:bin_num+1])
		return True
	except Exception as e:
		print(e)
		return False

@timeit
def test_equidepth():
	mu, sigma = 100, 15
	x = mu + sigma * np.random.randn(10000)
	print("\n** Equi-Depth Binning **\n")
	try:
		bin_num = 10
		tst = equidepth(x, bin_num, '-D')
		for _bin in tst:
			print(_bin[0:5])
		return True
	except Exception as e:
		print(e)
		return False

@timeit
def test_auto_hists():
	fn = "C:\\Users\\alber\\OneDrive - UTS\\UTS\\01.02\\[DAntcs] Data Analytics\\Ass2 - Analytics Data Set\\Excel Processing\\Intro Data Analysis.xlsx"
	sheet="Orig. Data"
	df = read_xlsx(fn, sheet)
	try:
		auto_run_hists(df.Geographic_Info2, "Ass2 XLSX Data", [1,5], '-F', '-D')
		return True
	except Exception as e:
		print(e)
		return False
