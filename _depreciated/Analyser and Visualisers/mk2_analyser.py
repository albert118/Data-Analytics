# Prototype analyser : MK2
# IDEA : take data "objects" and generate overview info on them
# graphically.
# SECONDARY : present quick stats on data "objects"
#
# Author : Albert Ferguson


import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

################################################################################
# Data Class object defs
################################################################################

class Data_Object:
	""" Object per row of dataset """
	_object_IDs = 0
	_field_dat = {}

	def __init__(self, *args, **kwargs):
		""" Pass field list through args, question user for types """
		self.fields = {}
		self.atts = []
		if 'fields' in kwargs and not Data_Object._field_dat:
			# data types need to be user clarified, retrieve and check 'em
			_fields = kwargs.pop('fields')
			self.atts = kwargs.pop('atts')
			for i in range(0, len(_fields)):
				print('\nField: {0}\tExample: {1}'.format(_fields[i], self.atts[i]))
				print("Enter type key: ") 
				self.fields[input()] = field

			# assign class specific field data
			Data_Object._field_dat = self.fields
		else:
			# data has already been added and confirmed in prev instance
			for field in _fields:
				self.fields = Data_Object._field_dat

		self.dimms = len(self.fields)
		self.id = Data_Object._object_IDs
		
		if 'id' in kwargs:
			Data_Object._object_IDs = kwargs.get('id')
		else:
			Data_Object._object_IDs += 1

	def lazy_debug(self):
		""" Print a full debug statement of the data object. """
		print("OBJECT - {}\n".format(self.id))

		for field in self.fields:
			print("FIELD: {0}\tTYPE: {1}}\n".format(self.field, self.fields.get(field)))

		# print("DIMENSIONS: {0}\n".format(self.dimms))
		return

	def add_att_data(self, dat, field):
		"""Add the data to the object """ 
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
		line_count = 0
		_objects = [] # objects to return

		# arg retrieval
		if 'delim' in kwargs:
			delim = kwargs.get('delims') 
		else: 
			delim = ','
		if 'dialect' in kwargs:
			_dialect = kwargs.get('dialect') 
		else:
			_dialect = 'excel'

		rdr = csv.reader(csvfile, delimiter = ' ', quotechar='|', dialect=_dialect)

		for row in rdr: # for object in rdr
			if line_count == 0:
				# get fields of data from row headers
				_fields = delim.join(row)
				_fields = _fields.split(delim)
				if '-v' in args: # verbose mode
					print("FILENAME:\t", file_name)
					print("FIELDS FOUND:")
					for elem in _fields:
						print(elem)
				line_count = 1
			else:
				if 'ezread' not in args:
					csv.QUOTE_NONNUMERIC

				_atts = delim.join(row)
				_atts = _atts.split(delim)
				_num_dimms = len(_atts) - 1 # removing row ID counter
				
				_new_object = Data_Object(fields=_fields, atts=_atts)
				for i in range(1, _num_dimms):
					_new_object.add_att_data(_atts[i], _fields[i])
				
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

def get_xlsx(file_name, sheet, *args, **kwargs):
	return pd.read_excel(file_name, sheet_name=sheet)

def write_xlsx(outs, *args, **kwargs):
	""" Write a list to an xlsx column """
	df = pd.DataFrame(outs)
	df.to_excel("output.xlsx")
	return

################################################################################
# Utility methods
################################################################################

def EWMA(inputs, *args, **kwargs):
	"""EWMA of an inputted list, returns the weighted list. """

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

	return weighted_list
