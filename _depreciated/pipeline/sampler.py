""" Demo the sampling methods in imblearn lib """
################################################################################
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours as ENN
from imblearn.ensemble import BalanceCascade
################################################################################

def sampler(df_orig, *args, **kwargs):
	# read the data and take note of the ids and dates
	if 'file' in args:
		df_orig = pd.read_csv("prep(1).csv")
	
	IDs = df_orig.Quote_ID
	target = df_orig.QuoteConversion_Flag
	data = df_orig.drop(['QuoteConversion_Flag'],axis=1).values

	# print our class distribution for the user to see... [class, count]
	# print("Before oversampling: ", sorted(Counter(target).items()))
	print("Before cascade: ", sorted(Counter(target).items()))

	# now use our SMOTE method of choice, either ENN or Tomeks to produce synthetic samples...


	####
	# ENN
	# SVM has better results with ENN
	####
	# enn = ENN(sampling_strategy="not majority", kind_sel="mode", n_neighbors=5, n_jobs=4, random_state=0)
	# smote_enn = SMOTEENN(enn=enn, random_state=0)
	# X_resampled, y_resampled = smote_enn.fit_resample(data, target)
	# print("SMOTE ENN: ", sorted(Counter(y_resampled).items()))

	####
	# Tomeks
	####
	# smote_tomek = SMOTETomek(random_state=0)
	# X_resampled, y_resampled = smote_tomek.fit_resample(data, target)
	# print("Using SMOTE: ", sorted(Counter(y_resampled).items()))

	fields = ['Quote_ID', 'Field_info1', 'Field_info2', 'Field_info3', 'Field_info4', 'Coverage_info1', 'Coverage_info2', 'Coverage_info3', 
	'Sales_info1', 'Sales_info2', 'Sales_info3', 'Sales_info4', 'Sales_info5', 'Personal_info1', 
	'Personal_info2', 'Personal_info3', 'Personal_info4', 'Property_info1', 'Property_info3', 
	'Property_info4', 'Property_info5', 'Geographic_info1', 'Geographic_info2', 'Geographic_info3', 
	'Geographic_info4', 'Geographic_info5']
	
	# synth = pd.merge(result, remain, on='Quote_ID', left_index=True)
	# if 'file' in args:
	# 	synth.to_csv("sampled(2).csv", index=False)
	
	bc = BalanceCascade(random_state=42)
	X_resampled, y_resampled = bc.fit_resample(data, target)
	print("Balanced Cascade: %s" % Counter(target[0]))

	data = pd.DataFrame(data = X_resampled, columns = fields)
	target = pd.DataFrame(data = y_resampled, columns = ['QuoteConversion_Flag'])
	# Now concat the data and target
	synth = pd.concat([target, data], axis=1)
	synth.Quote_ID = synth.Quote_ID.astype("int64")

	synth = pd.concat([target, data], axis=1)
	synth.Quote_ID = synth.Quote_ID.astype("int64")
	
	# synth = pd.merge(result, FIEL, on='Quote_ID', left_index=True)
	if 'file' in args:
		synth.to_csv("sampled(2).csv", index=False)
	
	return synth