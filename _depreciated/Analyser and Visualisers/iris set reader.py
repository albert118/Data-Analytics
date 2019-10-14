# IRIS DATA SET TESTING
# Three copies of the same data set, one doesnt work due to line-delimiter error 
# \r instead of \r\n

import csv
import matplotlib.pyplot as plt
import numpy as np
import math

################################################################################
# class def's
################################################################################

class Iris:
	""" The Iris class object """
	ID = 0
	def __init__(self, sep_len, sep_wid, pet_len, pet_wid, species):
		self.sepal_length = sep_len
		self.sepal_width = sep_wid
		self.petal_length = pet_len
		self.petal_width = pet_wid
		self.species = species
		self.id = Iris.ID
		Iris.ID += 1

	def print_iris(self):
		spec = "ID: [{id}] SPECIES: [{species}]".format(species=self.species, id=self.id)
		length = "LENGTH:\n\tPETAL: [{pet}]\n\tSEPAL: [{sep}]".format(pet=self.petal_length, sep=self.sepal_length)
		width = "WIDTH:\n\tPETAL: [{pet}]\n\tSEPAL: [{sep}]".format(pet=self.petal_width, sep=self.sepal_width)
		iris = '----------------------\n' +  spec + '\n' + length + '\n' + width + '\n----------------------'
		print(iris)
		return

################################################################################
# method def's
################################################################################

def get_data():
	print("Filename:\t", file1)
	name = 'C:\\Users\\Albert\\Desktop\\' + file1
	with open(name, newline = '') as csvfile:
		line_count = 0
		delim = ','
		iris_set = []
		rdr = csv.reader(csvfile, delimiter = ' ', quotechar='|', dialect='excel')
		for row in rdr:
			if line_count == 0:
				fields = delim.join(row)
				fields = fields.split(delim)
				print('FIELDS: ', fields)
				line_count += 1
			else:
				atts = delim.join(row)
				atts = atts.split(delim)
				new_iris = Iris(atts[1], atts[2], atts[3], atts[4], atts[5])
				iris_set.append(new_iris)
				line_count += 1
		return(iris_set)


def gen_histogram(inputs, **kwargs):
	""" 
	Generate a histogram of the input list data. 
	Possible kwargs:
		axis var name, index position
		title val

	"""
	

	# possible kwargs: axis/subplots name, axs coord
	arg_defaults = {
		'facecolor' : 'g',
		'alpha' : 0.5
		}
	# replace any parsed arg vals to override defaults
	for key, val in kwargs.items():
		if key in arg_defaults:
			arg_defaults[key] = val

	if 'axis' in kwargs.items():
		# get the axis object
		plots = axis['axis']
		# now get the index of the subplot
		plots[axis['row'], axis['col']].hist(inputs, **arg_defaults)
		if "title" in kwargs.keys():
			plots[axis['row'], axis['col']].set_title(kwargs['title'])
		else:
			plots[axis['row'], axis['col']].set_title("Probability Density Histogram of Data")

	else:
		plt.hist(inputs, **arg_defaults)
		# now test for global settings of 
		if "title" in kwargs.keys():
			plt.title(kwargs['title'])
		else:
			plt.title("Probability Density Histogram of Data")
		
	# binning, partition into equi-depth bins
	#  smooth by bin means, 
	#	replace every bin val with mean of bin
	#	or smooth by boundaries: leave lower and upper values (first and last)
	# 	then replace all other values by closest boundary (diff of value - boundary)
	# min-max, zscore, sigmoidal, etc... normalisations
	# sigmoidal used when outliers exist (or nonlinear). Using sigmoidal avoids stepwise
	# functions "weighting" outliers overly.
	plt.show() 
	return


############################## DATA FILES
file1 = 'iris-d12.csv'
file2 = 'iris-m1.csv'
file3 = 'iris-u2.csv'
############################## DATA FILES

fig, axs = plt.subplots(2, 2)
# fig.suptitle("Iris Test Visualisation")
# Testing and returning the data for printing
data = get_data()
seplen = []
sepwid = []
petwid = []
petlen = []

for iris in data:
	# iris.print_iris()
	seplen.append(iris.sepal_length)
	sepwid.append(iris.sepal_width)
	petwid.append(iris.petal_width)
	petlen.append(iris.petal_length)

# axs[0, 0].scatter(seplen, sepwid, c=(0,0,1))
# axs[0, 0].set_title('Scatter of Sepal Dimensions')
# axs[0, 1].scatter(petlen, petwid, c=(1,0,0))
# axs[0, 1].set_title('Scatter of Petal Dimensions')

# axs[1, 0].hist(seplen)
# axs[1, 0].set_title('Histogram of Sepal Lengths')

# axs[1, 1].hist(sepwid)
# axs[1, 1].set_title('Histogram of Sepal Widths')

# plt.show()

gen_histogram(seplen, title="Histogram of Sepal Lengths", axis={'axis':axs, 'row':0, 'col':0})

# TODO, add graphing and visualisation
#	Scatters
#	Frequency Curve(s)
#	Data Cubes??
