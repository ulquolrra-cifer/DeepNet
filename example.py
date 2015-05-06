
import sda
import numpy as np
from load_data import *
import nn2
from sklearn import preprocessing
if __name__ == '__main__':
	train_data,train_label = load_train('train.csv')
	test_data,ids = load_test('test.csv')
	train_label = train_label
	te=test_data[0:1000,:].copy()
	labels = form_label(train_label)
	min_max_scaler = preprocessing.MinMaxScaler()
	tr = min_max_scaler.fit_transform(train_data)
	te = min_max_scaler.fit_transform(test_data)
	mysda=sda.Sda([54,20,7])
	mysda.train(tr,20,100)
	mynn = nn2.NN([54,20,7])
	mynn.W['1'] = mysda.da['1'].W['1']
	mynn.nntrain(tr,labels,20,100)
	y=mynn.nnpredict(te)
	np.savetxt('predict.txt',y)
