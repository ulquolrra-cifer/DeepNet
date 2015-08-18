import numpy as np
import theano
import theano.tensor as T
from my_layers import Hiddenlayer,OutputsLayer,ConvPoolLayer
import time
from loaddata import *
from logistic_sgd import load_data
import pandas as pd
from collections import OrderedDict
def LeNet(learn_rate=0.01,batch_sizes=100,epochs=10,moment=0.0
,nkerns=[20,50],cv=5):
  #  datasets = load_data('mnist.pkl.gz')
  #  traindata,trainlabel = datasets[0]
    all_traindata,all_trainlabel = load_train('../data/train.csv')
  #  print "all traindata's num is %d" % (all_traindata.shape[0])
    if cv > 0 :
        sizes = all_traindata.shape[0]/cv
        traindata = all_traindata[0:(cv-1)*sizes,:]
        trainlabel = all_trainlabel[0:(cv-1)*sizes,:]
        validdata = all_traindata[(cv-1)*sizes:cv*sizes,:]
        validlabel = all_trainlabel[(cv-1)*sizes:cv*sizes,:]
   # print "traindata's num is %d,validdata's num is %d" % (traindata.shape[0],validdata.shape[0])
  #  train_m = traindata.shape[0]
  #  traindata = traindata.reshape((train_m,1,28,28))
    traindata = theano.shared(np.asarray(traindata,theano.config.floatX),borrow=True)
    trainlabel = theano.shared(np.asarray(trainlabel,theano.config.floatX),borrow=True)
    validdata = theano.shared(np.asarray(validdata,theano.config.floatX),borrow=True)
    validlabel = theano.shared(np.asarray(validlabel,theano.config.floatX),borrow=True)
    testdata = load_test('../data/test.csv')

    testdata = theano.shared(np.asarray(testdata,theano.config.floatX),borrow=True)
    x=T.matrix('x')
    y=T.matrix('y')
    index = T.lscalar('index')
    layer0_input = x.reshape((batch_sizes,1,28,28))
    layer0 = ConvPoolLayer(inputs=layer0_input,filter_shape=(nkerns[0],1,5,5),image_shape=(batch_sizes,1,28,28),poolsize=(2,2))
 #   def __init__(self,inputs,filter_shape,image_shape,rng=None,poolsize=(2,2)):
    layer1 = ConvPoolLayer(inputs=layer0.get_outputs(),filter_shape=(nkerns[1],nkerns[0],5,5),image_shape=(batch_sizes,nkerns[0],12,12),poolsize=(2,2))
    layer2_input = layer1.get_outputs().flatten(2)
    layer2 = Hiddenlayer(inputs=layer2_input,n_in=nkerns[1]*4*4,n_out=500,activation='tanh')
    layer3 = OutputsLayer(inputs=layer2.outputs(layer2_input),n_in=500,n_out=10,labels=y,activation='softmax')
    outputs = layer3.outputs(layer2.outputs(layer2_input)) 
    cost = layer3.cost(outputs,y)
    params = layer0.params+layer1.params+layer2.params+layer3.params
    gparams = []
    last_updates = {}
    for param in params:
        temp = np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX)
        last_updates[param] = theano.shared(temp)
    for param in params:
        gparam = T.grad(cost,param)
        gparams.append(gparam)
    updates = OrderedDict()
    for param,gparam in zip(params,gparams):
        weight_update = last_updates[param]
        upd = moment * weight_update - learn_rate * gparam
        updates[weight_update] = upd
        updates[param] = param + upd
    batch_num = traindata.get_value(borrow=True).shape[0]/batch_sizes
    batch_valid_num = validdata.get_value(borrow=True).shape[0]/batch_sizes
    #print "traindata's num is %d,validdata's num is %d,vathc_sizes is %d" % (batch_num,batch_valid_num,batch_sizes)
    #print "batch_valid_num is %d" % (batch_valid_num)
    train_model = theano.function([index],
                                    outputs=cost,
                                    updates=updates,
                                    givens={
                                               x:traindata[index*batch_sizes:(index+1)*batch_sizes],
                                               y:trainlabel[index*batch_sizes:(index+1)*batch_sizes]
                                           }
                                  )
    valid_model = theano.function([index],
                                    outputs=cost,
                                    givens={
                                                x:validdata[index*batch_sizes:(index+1)*batch_sizes],
                                                y:validlabel[index*batch_sizes:(index+1)*batch_sizes]
                                            }
                                  )
    predict_mode = theano.function([index],
                                    outputs=outputs,
                                    givens={
                                                x:testdata[index*batch_sizes:(index+1)*batch_sizes]
                                           }
                                   )
    best_valid_cost = np.inf
    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    epoch = 0
    done_loop = False
    valid_frequence = min(batch_num,patience/2)
    while (epoch < epochs) and (done_loop is False):
        epoch += 1
        t1 = time.time()
        mean_cost = []
        valid_cost = []
        for batch in xrange(int(batch_num)):
            mean_cost.append(train_model(batch))
            iter = (epoch - 1) * batch_num + batch
            if (iter + 1) % valid_frequence == 0 :

                for batch_val in xrange(int(batch_valid_num)):
                    valid_cost.append(valid_model(batch_val))
                mean_valid_cost = np.mean(valid_cost)
                if mean_valid_cost < best_valid_cost:
                    if mean_valid_cost < best_valid_cost * improvement_threshold:
                        patience = max(patience,iter*patience_increase)
                    best_valid_cost = mean_valid_cost
            if patience <= iter:
                done_loop = True
                break

        t2 = time.time()
        print '%d/%d,takes %f seconds,the traindata mean cost is %f,the bestvaliddata mean cost is %f' % (epoch,epochs,(t2-t1),np.mean(mean_cost),best_valid_cost)
    batch_test_num = testdata.get_value().shape[0]/batch_sizes
    predict = []       
    print 'predict the data!'
    for test_batch in xrange(batch_test_num):
        temp = predict_mode(test_batch)
        predict.append(temp)
    label = predict[0]
    for i in xrange(1,len(predict)):
        label = np.vstack((label,predict[i]))
    print 'predict is done!'
    label = np.argmax(label,1)
    id = range(1,28001)
    print 'writing to the file'
    o = pd.DataFrame(data={'ImageId':id,'Label':label})
    o.to_csv('predict.csv',index=False,quoting=3)
if __name__ == "__main__":
    LeNet()
