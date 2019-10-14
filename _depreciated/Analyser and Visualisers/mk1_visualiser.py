# Prototype analyser : MK1
# IDEA : take data "objects" and generate overview info on them
# graphically.
# SECONDARY : present quick stats on data "objects"
#
# Author : Albert Ferguson


import csv
import numpy as np
import matplotlib.pyplot as plt
import math

################################################################################
# Data Class object defs
################################################################################

class Data_Object:

	_object_IDs = 0
	_object_dimms = 0
	_dimms_has_been_set = False

	def __init__(self, filename, dimms, field_data, ):
		
		# set metadata
		self.object_ID = Data_Object._object_IDs
		
		if not _dimms_has_been_set:
			Data_Object._object_dimms = dimms
			Data_Object._dimms_has_been_set = True

		# 'field':'type'
		self.fields = field_data
		self.object_source = filename
		
		self.get_quick_stats()
		Data_Object._object_IDs += 1
		
		return

	def lazy_debug(self):
		""" Print a full debug statement of the data object. """
		print('OBJECT - %s\n' %self.object_ID)

		for field in self.fields:
			print('FIELD: %s\tTYPE: %s\n' %field %self.fields[field])

		print('DIMENSIONS: %s\n' %self.num_dimms)
		print('SOURCED FROM: %s\n', %self.object_source)

		return

	def get_dimms(self):
		return(Data_Object._object_dimms)

################################################################################
# Data method defs
################################################################################

def gen_prob_histo(objects):
	""" Generate a CDF probability histogram of the data set. """
	pass
	return

def gen_freq_histo(objects):
	""" Generate a culminative frequency histogram of the data set. """
	pass
	return

def gen_biv_scatt(objects):
	""" Generate a bivariate scatter plot of the data set 'x' and 'y' values. """
	pass
	return

def get_quick_stats(data_in, *args, **kwargs):
	""" Get quick statistics of the data set. """
	if '-v' in args:
		_verbose = True
	else:
		_verbose = False

	if 'type' in kwargs:
		_type = kwargs.get('type')
	else:
		# default type
		_type = 'int'

	_field_stat_lst = []
	i = objects.Data_Object.get_dimms()
	
	for field in objects
	_stats = {}

	np.std()
	# AVOID NaN data!	
	# Sample SD
	# min, max
	# median
	# mean
	# mode
	
	return



################################################################################
# Utility methods
################################################################################

def get_csv(file_name, *args, **kwargs):
	""" 
	Open a CSV object of name file_name. 
	Optional args,
		 -v for verbose mode.
	Optional kwargs,  
		delims - set delimiter char
		dialect - set csv reader dialect
	
	Returns a tuple <fields, objects>
	Where fields is a list of all data field names
	where objects is a list of ordered attribute lists
	NOTE: each attribute is indexed according to appearance in fields list
	"""	

	with open(file_name, newline = '') as csvfile:
		
		if 'delim' in kwargs:
			delim = kwargs.get('delims')
		else:
			delim = ','

		if 'dialect' in kwargs:
			_dialect = kwargs.get('dialect')
		else:
			_dialect = 'excel'

		rdr = csv.reader(csvfile, delimiter = ' ', quotechar='|', dialect=_dialect)
		
		line_count = 0
		_objects = [] # objects to return

		for row in rdr:
			if line_count == 0:
				# get fields of data from row headers
				_fields = delim.join(row)
				_fields = _fields.split(delim)
				if '-v' in args: # verbose mode
					print("FILENAME:\t", file_name)
					print("FIELDS FOUND:\n")
					for elem in _fields:
						print(elem)
				line_count = 1
			else:
				_atts = delim.join(row)
				_atts = _atts.split(delim)
				_num_dimms = len(_atts) - 1 # removing row ID counter
				
				_new_object = []
				for i in range(1, _num_dimms):
					_new_object.append(_atts[i])
				
				_objects.append(_new_object)
				line_count += 1

		return _fields, _objects

def get_file_type(f_name):
	""" determine filetype from filename """
		f_name = self.object_source
		f_type = ""
		i = 1
		while f_name[-i] is not ".":
			f_type = f_name[-i] + f_type
			i += 1
			if i == len(f_name)+1:
				break

		return f_type