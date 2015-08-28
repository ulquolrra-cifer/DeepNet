import numpy as np
from loaddata import *
import nn_RMSprop
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
if __name__ == '__main__':
	train_data,train_label = load_train('../data/train.csv')
	test_data = load_test('../data/test.csv')
	mynn = nn_RMSprop.NN_RMSprop([784,150,10])
	mynn.nntrain(train_data,train_label,50,10)
	
	y=mynn.nnpredict(test_data)
	predict = np.argmax(y,1)
	id = range(1,28001)
	output = pd.DataFrame( data={"ImageId":id, "Label":predict} )
	output.to_csv( "predict.csv", index=False, quoting=3 )
