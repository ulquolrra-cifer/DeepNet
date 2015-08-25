import numpy as np
import hessianff as hf
import pandas as pd
from loaddata import *
import cPickle
def test():
    traindata,trainlabel = load_train('../data/train.csv')
    testdata = load_test('../data/test.csv')
    ff = hf.NN([784,100,10])
    ff.run_hf(traindata,trainlabel,epochs = 2000,iters=250)
#    cPickle.dump(ff,open("data.pkl","wb"))
    predict = ff.predict(testdata)
    np.savetxt('predict.txt',predict)
    predict = np.argmax(predict,1)
    id = range(1,28001)
    output = pd.DataFrame( data={"ImageId":id, "Label":predict} )
    output.to_csv( "predict.csv", index=False, quoting=3 )

if __name__ == '__main__':
    test()
