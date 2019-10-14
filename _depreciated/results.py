import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import randint
from sklearn.model_selection import train_test_split

################################################################################
# Util methods

def clear():
	os.system("clear")
	os.system("cls")
	return

def get_fields(df):
	fields=[]
	for field in df:
		fields.append(field)
	
	print(df.shape)
	return fields

def chk_NaN(df, fields):
	NaNs=[]
	# get NaN/None/NA/Null checks per dimension
	for field in fields:
		NaNs.append(df[field].isna().sum())
	return NaNs

################################################################################

# data overview and sanity checks
df = pd.read_csv("Assignment3_TrainingSet(2).csv")


x = df['Personal_info5'].isna().sum()
missing_percentage = (x/objects) * 100
# 47.349 %
df = df.drop(['Personal_info5'], axis=1)
fields.remove("Personal_info5")

# checking unique values of Geo3
unique_vals = []
for elem in df['Geographic_info3']:
	if elem not in unique_vals:
		unique_vals.append(elem)
# finds 2, [-1, 25] => binarise in tmp csv following...

unique_vals = []
for elem in df['Personal_info1']:
	if elem not in unique_vals:
		unique_vals.append(elem)
#>>> unique_vals
# ['N', 'Y', nan]
# checking count of X, if majority then set NaN to 0, else 1

# Property_info1  - x:  68074, 0 is majority set NaN = 0
# Personal_info1  - x:  77821, 0 is majority set NaN = 0
# Personal_info4  - x:  78169, 0 is majority set NaN = 0
# Property_info2  - x:  78225, all vals==0 drop this field
chk_fields = ['Property_info1', 'Personal_info1', 'Personal_info4', 'Property_info2']
for field in chk_fields:
	x=0
	for elem in df[field]:
		if elem=="N" or elem==0:
			x+=1
	print(field, ' - x: ', x)

# save info so far to tmp csv, in this csv we now should edit fields from strings to binaries
# and append these rows onto the tmp to be re read
df.to_csv("predata(1).csv")

# re read now that csv edits completed in Excel...
df = pd.read_csv("predata(1).csv")
# then drop Prop2 as all vals=0 and other irrelevant/preprocessed rows
df = df.drop(['Unnamed: 0', 'Geographic_info3',"Property_info2"], axis=1)

# Get fields and get info on df
fields = []
objects, dims = df.shape

for field in df:
	fields.append(field)

# get NaN/None/NA/Null checks per dimension on new doc
Nan=0
for field in fields:
	Nan+=df[field].isna().sum()
	# if Nan ever goes over zero, call next line
	if Nan > 0:
		# Fixing any remaining null values in set, most removed by binarisation 
		print(field, ' Nan count: ', Nan)
		df = df.astype(object).replace(np.nan, 0)
		break
# ALL 0!!! Great!! Don't have to call next line then...


# remove any ',' chars from thousands separator....
df['Field_info3']=df['Field_info3'].str.replace(",","").astype(float)

# before we go any further, drop any fields with 0 variance (constant values across all entries...)
for field in fields:
	if df[field].var()==0:
		fields.remove(field)
		df=df.drop([field],axis=1)
		print("Dropped: ", field)

# fields to standardise...
scale_fields=['Sales_info5', 'Field_info3', 'Property_info5', 'Property_info5', 'Geographic_info1', 'Geographic_info2']

for dimension in scale_fields:
	df[dimension]=pre.scale(df[dimension].values, copy=False)

# these fields are to be remapped from their numerical (2 var vals) to bin [1,0]..simply MinMax them
num_bin_fields=[]
# fields to MinMax scaled...
scale_fields=['Coverage_info1', 'Coverage_info3_NUM','Coverage_info2','Sales_info2', 'Sales_info3', 'Personal_info2', 'Personal_info3_NUM']
# fields to encode to numerical equivalents, in order of appearance then num=idx+1 from list of unique val's..
label_fields=['Property_info3', 'Sales_info4', 'Coverage_info3', 'Field_info1', 'Personal_info3','Property_info3','Geographic_info5', 'Field_info4', 'Property_info1','Geographic_info4']

le = pre.LabelEncoder()
for dimension in label_fields:
	le.fit(df[dimension])
	df[dimension]= le.transform(df[dimension])

# now add label fields to scale fields, MinMax scale these all then...
# first remove the binaries, no point MinMax'ing them
rem=['Field_info4', 'Property_info1','Geographic_info4']
for elem in rem:
	label_fields.remove(elem)

for elem in label_fields:
	scale_fields.append(elem)

scaler = pre.MinMaxScaler(copy=False)
for dimension in scale_fields:
	scaler.fit(df[dimension].values.reshape(-1,1))
	df[dimension]=scaler.transform(df[dimension].values.reshape(-1,1))


# TODO Property_info3, Sales_info4, (str -> num -> MinMax)
df.to_csv("predata(2).csv")
df = pd.read_csv("predata(2).csv")
df = df.drop(['Unnamed: 0'], axis=1)
# Get fields and get info on df
fields = []
objects, dims = df.shape

for field in df:
	fields.append(field)

# Models that must be used "tree", 'KNN', 'Random Forest', 'SVM', 'ANN'

# Prep input data for model use (2D array of data)
# data = df_new.as_matrix(columns=df_new.columns[2:])
# for feature in data:
# 	feature=feature.transpose()

# Now get the data vals of f(x)=y, where y is our QuoteConversionFlag and X is remaining fields
y = df['QuoteConversion_Flag'].values
X = df.drop(["QuoteConversion_Flag",'Quote_ID', 'Original_Quote_Date'], axis=1).values

clf=svm.SVC(kernel="linear")
anova_filter= SelectKBest(f_regression, k=5)
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])

anova_svm.set_params(anova__k=20, svc__C=.1).fit(X, y)

prediction = anova_svm.predict(X)
anova_svm.score(X, y)
f1_score(y, prediction, average='macro')
f1_score(y, prediction, average=None)

anova_svm['anova'].get_support()

# Checking Y vector...
# number of 1s=x and y=0s
x=14910
y=78225
x/y # ... 0.1906040268456376
# 19.06% of data set is positive conversion, set is skewed!! Superior sampling method is now required!

# 57% variance, theory dropping binary data will yield better variance (drop lowest data density...)
X = df.drop(['Quote_ID', 'Original_Quote_Date', 'QuoteConversion_Flag', 'Property_info1_bin', 'Geographic_info4_bin', 'Geographic_info3_bin', 'Field_info1_J', 'Field_info1_E', 'Field_info1_B', 'Field_info1_F', 'Field_info1_C', 'Field_info1_K', 'Field_info1_D','Sales_info5'], axis=1).values
# 47% variance, baseline
X = df.drop(['Quote_ID', 'Original_Quote_Date', 'QuoteConversion_Flag'], axis=1).values
# 47% variance, test with no binary, follow on from 57% test
X = df.drop(['Quote_ID', 'Original_Quote_Date', 'QuoteConversion_Flag', 'Field_info4_bin', 'Personal_info1_bin', 'Property_info1_bin', 'Geographic_info4_bin', 'Geographic_info5_TX', 'Geographic_info5_NJ', 'Geographic_info5_CA', 'Geographic_info5_IL', 'Geographic_info3_bin', 'Field_info1_J', 'Field_info1_E', 'Field_info1_B', 'Field_info1_F', 'Field_info1_C', 'Field_info1_K', 'Field_info1_D'], axis=1).values
# theory, drop highest variance rows (and zero) from df for X
X = df.drop(['Quote_ID', 'Original_Quote_Date', 'QuoteConversion_Flag', 'Field_info3', 'Sales_info5', 'Property_info5', 'Geographic_info1', 'Geographic_info2', 'Geographic_info5_CA','Geographic_info3_bin'], axis=1).values
# 40.8% variance

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df['QuoteConversion_Flag']], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g']
targets = [0,1]
for target, color in zip(targets,colors):
	indicesToKeep = finalDf['QuoteConversion_Flag'] == target
	ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

ax.legend(targets) 
ax.grid()

pca.components_
sum(pca.explained_variance_ratio_)

plt.show()