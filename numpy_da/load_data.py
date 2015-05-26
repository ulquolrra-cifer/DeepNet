import numpy as np
import pandas as pd
from pandas import DataFrame
def mapminmax(data):
	data=np.mat(data)
	m,n=np.shape(data)
	train_data=np.mat(np.zeros((m,n)))
	for i in range(n):
		cols=data[:,i]
		max_data=np.max(cols)
		min_data=np.min(cols)
		train_data[:,i]=(cols-min_data)/(max_data-min_data)
	return train_data
def form_label(label):
	m=np.shape(label)[0]
	l=np.zeros((m,7))
	for i in range(m):
		if label[i]==1:
			l[i][0]=1
		elif label[i]==2:
			l[i][1]=1
		elif label[i]==3:
			l[i][2]=1
		elif label[i]==4:
			l[i][3]=1
		elif label[i]==5:
			l[i][4]=1
		elif label[i]==6:
			l[i][5]=1
		elif label[i]==7:
			l[i][6]=1
	return l
	
def load_train(filename):
    train = pd.read_csv(filename,sep=',')
    m,n = train.shape
    train_array = np.zeros((m,n-2))
    labels = np.zeros((m,1))
    values = train.columns.values
    for i in range(1,n):
        col_val = values[i]
        if i == n-1:
            labels = train[col_val]
        else:
            train_array[:,i-1] = train[col_val]
    
    return train_array,np.array(labels)

def load_test(filename):
    test = pd.read_csv(filename,sep=',')
    m,n = test.shape
    test_array = np.zeros((m,n-1))
    ids = np.zeros((m,1))
    values = test.columns.values

    for i in range(n):
        col_val = values[i]
        if i==0:
			ids=test[col_val]
        else:
            test_array[:,i-1] = test[col_val]
    return test_array,np.array(ids)
