import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def graph(df):
	fig = plt.figure(figsize = (8,8))
	ax = fig.add_subplot(1,1,1)
	ax.set_ylabel('Principal Component 2', fontsize = 15)
	ax.set_xlabel('Principal Component 1', fontsize = 15)
	ax.set_title('SMOTE with Tomek Links submethod', fontsize = 20)
	colors = ['g', 'r']
	classes = [0, 1]

	for target, color in zip(classes, colors):
		indicesToKeep = df['QuoteConversion_Flag'] == target
		ax.scatter(df.loc[indicesToKeep, 'principal component 1'], df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

	ax.legend(classes)
	ax.grid()
	plt.show()

	return