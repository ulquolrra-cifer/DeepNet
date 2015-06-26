
import sda
import numpy as np
from loaddata import *
import nn2
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
if __name__ == '__main__':
	train_data,train_label = load_train('../data/train.csv')
	test_data = load_test('../data/test.csv')
	#test_data = load_test('test.csv')
#	pca=PCA(n_components=0.9,whiten=True)
#	x=pca.fit(test_data).transform(test_data)
#	train_data = train_data 
#	test_data = test_data / 255.0
	mysda=sda.Sda([784,150,10],activation_function='sigm',learningRate = 2,momentum = 0.5,weightPenaltyL2 = 0,sparsityparameter = 0,beta=0.5,inputzeroMaskedFraction=0,output = 'sigm',jacobi_penalty = 0,scaling_learningRate = 0.99,dropout = 0.3)
	mysda.train(train_data,50,1)
#	mysda.train(test_data,50,1)
	mynn = nn2.NN([784,150,10],activation_function='sigm',learningRate = 1,momentum = 0.5,weightPenaltyL2 = 0,sparsityparameter = 0,beta=0.5,inputzeroMaskedFraction=0,output = 'softmax',jacobi_penalty = 0,scaling_learningRate = 0.99,dropout = 0)
	mynn.W['1'] = mysda.da['1'].W['1']
	mynn.b['1'] = mysda.da['1'].b['1']
	mynn.W['2'] = mysda.da['2'].W['1']
	mynn.b['2'] = mysda.da['2'].b['1']
#	mynn.W['3'] = mysda.da['3'].W['1']
#	mynn.b['3'] = mysda.da['3'].b['1']
	mynn.nntrain(train_data,train_label,50,1)
	
	y=mynn.nnpredict(test_data)
	predict = np.argmax(y,1)
	id = range(1,28001)
	output = pd.DataFrame( data={"ImageId":id, "Label":predict} )
	output.to_csv( "predict.csv", index=False, quoting=3 )
