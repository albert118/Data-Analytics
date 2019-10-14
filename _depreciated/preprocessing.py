""" 
Preprocess the data set!!
-------------------------
	Remove ',' as thousands separator
	Drop rows with approx. 50% missing data
	Drop low variance rows from set
	Standardise continous value fields
	Transform string categoricals to numeric fields
	Normalise categorical fields
	Transform string binaries (Y, N) to 0,1
-------------------------
Output the lot to a new CSV with updated dimensions!!
"""

################################################################################
# Imports
import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
################################################################################


################################################################################
# data overview and sanity checks
df = pd.read_csv("Assignment3_TrainingSet(2).csv")
# Prop2 is constant all==0, Pers5 is approx. 50% data missing...
df = df.drop(['Personal_info5',"Property_info2"], axis=1)
# Get fields and get info on df
fields = []
objects, dims = df.shape
for field in df:
	fields.append(field)
################################################################################



################################################################################
# Scaling
# remove any ',' chars from thousands separator in Field3....
df['Field_info3']=df['Field_info3'].str.replace(",","").astype(float)

# fields to standardise...
scale_fields=['Sales_info5', 'Field_info3', 'Property_info5', 'Geographic_info1', 'Geographic_info2']

for dimension in scale_fields:
	df[dimension]=pre.scale(df[dimension].values, copy=False)
print("Scaled...")

# fields to MinMax scaled...
scale_fields = ['Coverage_info1', 'Coverage_info3','Coverage_info2','Sales_info2', 'Sales_info3', 'Personal_info2', 'Personal_info3']
# fields to encode to numerical equivalents, in order of appearance then num=idx+1 from list of unique val's..
label_fields = ['Personal_info1', 'Property_info3', 'Sales_info4', 'Coverage_info3', 'Field_info1', 'Personal_info3','Property_info3','Geographic_info5','Field_info4', 'Property_info1', 'Geographic_info3', 'Geographic_info4']
# Binary fields are label fields, but we don't want to waste time MinMax normalising these as they'll either be Min or Max anyway!!
binary_fields = ['Field_info4', 'Property_info1','Geographic_info4', 'Personal_info1']

# NaN checks before..
print("NaN sum:\n", df.isna().sum())
# from prev analysis, majority is N is binaries, for ~20-40 missing samples, replace with 'N' for 'No'
df = df.astype(object).replace(np.nan, "N")
# NaN checks after...
print("NaN sum:\n", df.isna().sum())
input()

# Now lets convert all string fields, incl. binaries, to numerical categories..
le = pre.LabelEncoder()
for dimension in label_fields:
	le.fit(df[dimension])
	df[dimension]= le.transform(df[dimension])
print("Categories fitted...")

# remove the binaries so they dont get MinMax'd
for elem in binary_fields:
	label_fields.remove(elem)

# now add label fields to scale fields, MinMax scale these all then...
for elem in label_fields:
	scale_fields.append(elem)
print("Readying categories for numerical conversion...")

# apply MinMax scaling to remaining fields..
scaler = pre.MinMaxScaler(copy=False)
for dimension in scale_fields:
	scaler.fit(df[dimension].values.reshape(-1,1))
	df[dimension]=scaler.transform(df[dimension].values.reshape(-1,1))
print("MinMax scaling completed...")
################################################################################




################################################################################
# Checking low variance fields...

# before we go any further, drop any fields with 0 variance (constant values across all entries...)
# prior, drop columns such as date, that will not be able to calculated for variance correctly
dates=df.Original_Quote_Date
ID=df.Quote_ID
df=df.drop(['Quote_ID', 'Original_Quote_Date'],axis=1)
print("Dropped ID and dates...")
fields=[]
for field in df:
	fields.append(field)

for field in fields:
	if df[field].var()==0:
		print("Checking variance on: ")
		print(field)
		input()
		fields.remove(field)
		df=df.drop([field],axis=1)
		print("Dropped: ", field)
print("No variance issues, all fields validated...")
################################################################################



################################################################################
# Results!
df = pd.concat([ID, dates, df], axis = 1)
df.to_csv("preprocessing(1).csv")