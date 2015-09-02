import numpy as np
import pandas as pd
from pandas import DataFrame
#def form_label(labels):
#	l=np.zeros((np.shape(labels)[0],10))
#	for i in range(np.shape(labels)[0]):
#		if labels[i]==0:
#			l[i,0]=1
#		elif labels[i]==1:
#			l[i,1]=1
#		elif labels[i]==2:
#			l[i,2]=1
#		elif labels[i]==3:
#			l[i,3]=1
#		elif labels[i]==4:
#			l[i,4]=1
#		elif labels[i]==5:
#			l[i,5]=1
#		elif labels[i]==6:
#			l[i,6]=1
#		elif labels[i]==7:
#			l[i,7]=1
#		elif labels[i]==8:
#			l[i,8]=1
#		elif labels[i]==9:
#			l[i,9]=1
#	return l
def form_label(labels):
    results = np.zeros((labels.shape[0],10))
    results[xrange(labels.shape[0]),labels] =1.
    return results
def load_train(filename):
	train = pd.read_csv(filename,sep=',')
	m,n = train.shape
	train_array = np.zeros((m,n-1))
	labels = np.zeros((m,1))
	values = train.columns.values
	for i in range(0,n):
		col_val = values[i]
		if i == 0:
			labels = train[col_val]
		else:
			train_array[:,i-1] = train[col_val]
    
	return train_array/255.0,form_label(labels)
def load_test(filename):
	test = pd.read_csv(filename,sep=',')
	m,n = test.shape
	test_array = np.zeros((m,n))
	values = test.columns.values
	for i in range(n):
		col_val = values[i]
		test_array[:,i] = test[col_val]
	return test_array/255.0
