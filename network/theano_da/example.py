#from __future__ import division
from loaddata import *
import sda1
from nn import ForwordNN
import nn
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from my_layers import Hiddenlayer,OutputsLayer,form_label,unfold
import time
if __name__ == "__main__":
    traindata,trainlabel = load_train('../data/train.csv')
    y_pre=np.argmax(trainlabel,1)
    testdata = load_test('../data/test.csv')
    learning_rate = 0.5
    batch_sizes = 20
    pre_epochs = 10
    fine_epochs = 50
    drop = ([0.5,0.5],[0.5,0.5])
    moment = 0.5
    x = T.matrix('x')
    y = T.matrix('y')
    index = T.lscalar()
    test_data = T.matrix('test_data')
    numpy_rng = np.random.RandomState(123)
    traindata = theano.shared(np.asarray(traindata,dtype=theano.config.floatX),borrow=True)
    trainlabel = theano.shared(np.asarray(trainlabel,dtype=theano.config.floatX),borrow=True)
    testdata = theano.shared(np.asarray(testdata,dtype=theano.config.floatX),borrow=True)
    batch_num = traindata.get_value(borrow=True).shape[0] / batch_sizes
    mysda = sda1.Sda(n_in=784,n_out=10,hidden_sizes=[500,500],np_rng=numpy_rng,dropout=drop)
#	pre_w1 = mysda.params[0].get_value()
#	pre_b1 = mysda.params[1].get_value()
    mynn=ForwordNN(inputs=x,label=y,n_in=784,n_out=10,hiddensizes=[500,500],rng=numpy_rng,activation='sigmoid',output_activation='softmax')
    print 'pretraining the model!'
    pre_functions = mysda.pretraining(inputs=traindata,batch_size=batch_sizes,learning_rate=learning_rate,denoising=[0.0,0.0,0.0])

    for i in xrange(mysda.n_layers):

        for num_epochs in xrange(pre_epochs):
            c=[]
            t1=time.time()
            for n_batchs in xrange(batch_num):
                c.append(pre_functions[i](n_batchs))
            t2=time.time()
            print '%d layers pretraining, %d/%d taken %f seconds and the mean cost is %f' % (i,num_epochs,pre_epochs,(t2-t1),np.mean(c))

    print 'finetune trainging the model!'
    mynn=unfold(mysda,mynn,drop)
    min_batch_cost = []
    cost = mynn.outlayer.cost(mynn.predict_outputs(x),y)+mynn.L2_reg*mynn.L2+mynn.L1_reg*mynn.L1
    gparams =[]
    for param in mynn.params:
        gparam=T.grad(cost,param)
        gparams.append(gparam)
    updates = {}
    for param,gparam in zip(mynn.params,gparams):
        weight_update = mynn.updates[param]
        upd = moment*weight_update-learning_rate*gparam
        mynn.updates[param] = upd
        updates[param] = param + upd
#	cost,update = mynn.get_cost_update(x,y,learning_rate)
    train_nn = theano.function([index],
                cost,
                updates = updates,
                givens = {
                            x:traindata[index*batch_sizes:(index+1)*batch_sizes,:],
                            y:trainlabel[index*batch_sizes:(index+1)*batch_sizes,:]
                        }
                )
    for num_epochs in range(fine_epochs):
        t1=time.time()
        for num_batch in xrange(batch_num):
            min_batch_cost.append(train_nn(num_batch))
        t2=time.time()
        print 'The %d/%dth training,takes %f seconds,cost is %f' %(num_epochs+1,fine_epochs,(t2-t1),np.mean(min_batch_cost))
    predict_nn = theano.function([],outputs=mynn.predict_outputs(test_data),givens={test_data:testdata})
    predict_label = predict_nn()
#	np.savetxt('output.txt',predict_label)
    y = np.argmax(predict_label,1)
#	accuraty = y-y_pre
#	a= len(accuraty[accuraty[accuraty==0]])
#	print a
    id = range(1,28001)
    output = pd.DataFrame( data={"ImageId":id, "Label":y} )
    output.to_csv( "predict.csv", index=False, quoting=3 )
