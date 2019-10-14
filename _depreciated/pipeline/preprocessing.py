import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
################################################################################


################################################################################
# data overview and sanity checks

def prep(*args, **kwargs):
	df = pd.read_csv("Assignment3_TrainingSet(2).csv")
	ID=df.Quote_ID
	# Prop2 is constant all==0, Pers5 is approx. 50% data missing...
	df = df.drop(['Personal_info5',"Property_info2", 'Quote_ID', 'Original_Quote_Date'], axis=1)
	# Get fields and get info on df
	fields = []
	objects, dims = df.shape
	for field in df:
		fields.append(field)

	# remove any ',' chars from thousands separator in Field3....
	df['Field_info3']=df['Field_info3'].str.replace(",","").astype(float)

	# fields to encode to numerical equivalents, in order of appearance then num=idx+1 from list of unique val's..
	label_fields = ['Personal_info1', 'Property_info3', 'Sales_info4', 'Coverage_info3', 'Field_info1', 'Personal_info3','Property_info3','Geographic_info5','Field_info4', 'Property_info1', 'Geographic_info3', 'Geographic_info4']
	# Binary fields are label fields, but we don't want to waste time MinMax normalising these as they'll either be Min or Max anyway!!
	binary_fields = ['Field_info4', 'Property_info1','Geographic_info4', 'Personal_info1']

	# NaN checks before..
	#print("NaN sum:\n", df.isna().sum())
	# from prev analysis, majority is N is binaries, for ~20-40 missing samples, replace with 'N' for 'No'
	df = df.astype(object).replace(np.nan, "N")
	# NaN checks after...
	#print("NaN sum:\n", df.isna().sum())

	# Now lets convert all string fields, incl. binaries, to numerical categories..
	le = pre.LabelEncoder()
	for dimension in label_fields:
		le.fit(df[dimension])
		df[dimension]= le.transform(df[dimension])
	#print("Categories fitted...")

	# Checking low variance fields...
	# before we go any further, drop any fields with 0 variance (constant values across all entries...)	
	for field in fields:
		if df[field].var()==0:
			print("Checking variance on: ")
			print(field)
			fields.remove(field)
			df=df.drop([field],axis=1)
			print("Dropped: ", field)
	#print("No variance issues, all fields validated...")

	# Results!
	df = pd.concat([ID, df], axis = 1)
	if 'file' in args:
		df.to_csv("prep(1).csv", index=False)

	return df