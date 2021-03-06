import theano
import theano.tensor as T
import numpy as np
from my_layers import Hiddenlayer,OutputsLayer
import time
from loaddata import *
import pandas as pd
from collections import OrderedDict
mode = theano.Mode(linker='cvm')
class ForwordNN(object):
    def __init__(self,inputs,label,n_in,n_out,hiddensizes,L1_reg = 0.00,L2_reg=0.00,activation='sigmoid',output_activation='sigmoid',rng=None):
        self.inputs = inputs
        self.label = label
        self.n_in = n_in
        self.n_out = n_out
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        if rng is None:
            self.rng = np.random.RandomState()
        self.len_hid = len(hiddensizes)
#		self.w = []
        self.sizes = [self.n_in]+hiddensizes+[self.n_out]
        self.layer = []
        self.params = []
#		self.hiddensizes = hiddensizes
        input_temp = self.inputs
        for i in range(len(hiddensizes)):
            layer = Hiddenlayer(input_temp,self.sizes[i],self.sizes[i+1],activation=activation)
            self.params += layer.params
            self.layer.append(layer)
            input_temp = layer.output
        self.outlayer = OutputsLayer(self.layer[-1].output,self.sizes[-2],self.sizes[-1],self.label,activation = output_activation)
        self.output=self.outputs(self.inputs)
        self.params += self.outlayer.params
        self.L2 = 0
        self.L1 = 0
#		self.L2 = ((self.layer[0].w**2).sum()+(self.outlayer.w**2).sum())
        for i in range(len(self.layer)):
            self.L2 += (self.layer[i].w**2).sum()
            self.L1 += abs(self.layer[i].w).sum()
        self.L2 += (self.outlayer.w**2).sum()
        self.L1 += abs(self.outlayer.w).sum()
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX)
            self.updates[param]=theano.shared(init)
    def outputs(self,inputs):
#		for i in range(len(self.layer)):
        outputs = self.outlayer.outputs(inputs)
        return outputs
    def predict_outputs(self,inputs):
        temp_inputs = inputs
        for i in xrange(len(self.layer)):
            temp_outputs = self.layer[i].outputs(temp_inputs)
            temp_inputs = temp_outputs
        outputs = self.outlayer.outputs(temp_inputs)
        return outputs
    def get_cost_update(self,inputs,label,learning_Rate):
        y = self.outputs(inputs)
        self.label = label
        self.cost = self.outlayer.cost(y,self.label)
        dparams = [T.grad(self.cost,params) for params in self.params]
        updates = [(param,param - learning_Rate*dparam)
                    for param,dparam in zip(self.params,dparams)
        ]
        return self.cost,updates

def train(inputs,labels,n_in,n_out,learning_Rate,hidden_sizes,batch_sizes,epochs,moment=0.5,nn=None):
    m = np.shape(inputs)[0]
    n_train_batchs = m / batch_sizes
    train_data = theano.shared(np.asarray(inputs,dtype=theano.config.floatX),borrow=True)
    train_label = theano.shared(np.asarray(labels,dtype=theano.config.floatX),borrow=True)
#	train_label = T.cast(train_label,'float64')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')
    min_batch_cost = []
    if nn is None:
        mynn = ForwordNN(x,y,n_in,n_out,hidden_sizes)
    else:
        mynn=nn
	#	cost,update = mynn.get_cost_update(x,y,learning_Rate)
    cost = mynn.outlayer.cost(mynn.predict_outputs(x),y)+mynn.L2_reg*mynn.L2+mynn.L1_reg*mynn.L1
#	lr = T.scalar('lr',dtype=theano.config.floatX)
#	mom = T.scalar('mom',dtype=theano.config.floatX)
    gparams =[]
    for param in mynn.params:
        gparam=T.grad(cost,param)
        gparams.append(gparam)
#    gparams_mom = []
#    for param in mynn.params:
           
    updates = OrderedDict()
#   for gparam_mom,gparam in zip(mynn.mom,gparams):
 #       updates[gparam_mom] = moment*gparam + (1-moment)*learning_Rate*gparam

  #  for param,gparam_mom in zip(mynn.params,mynn.mom):
   #     step_param = param + updates[gparam_mom]
    #    updates[param]= step_param
    for param,gparam in zip(mynn.params,gparams):
        weight_update = mynn.mom[param]
        upd = moment*weight_update-learning_Rate*gparam
        updates[weight_update] = upd
        updates[param] = param + upd
    train_nn = theano.function([index],
                cost,
                updates = updates,
                givens = {
                            x:train_data[index*batch_sizes:(index+1)*batch_sizes,:],
                            y:train_label[index*batch_sizes:(index+1)*batch_sizes,:]
                         },
                mode=mode
                )
    for num_epochs in range(epochs):
        t1=time.clock()
        for num_batch in xrange(n_train_batchs):
            min_batch_cost.append(train_nn(num_batch))
        t2=time.clock()
        print 'The %d/%dth training,takes %f seconds,cost is %f' %(num_epochs+1,epochs,(t2-t1),np.mean(min_batch_cost))
    return mynn

#def example_nn(inputs,labels):
def fine_train(nn,datasets,learning_Rate,batch_sizes,epochs):
	train_set_x, train_set_y = datasets[0]
	n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_sizes
	
	train_label = T.cast(train_label,'float64')
	index = T.lscalar()
	x = T.matrix('x')
	y = T.matrix('y')
	min_batch_cost = []
	if nn is None:
		mynn = ForwordNN(x,y,n_in,n_out,hidden_sizes)
	else:
		mynn=nn
	cost,update = mynn.get_cost_update(x,y,learning_Rate)
	train_nn = theano.function([index],
				cost,
				updates = update,
				givens = {
							x:train_data[index*batch_sizes:(index+1)*batch_sizes,:],
							y:train_label[index*batch_sizes:(index+1)*batch_sizes,:]
						}
				)
	for num_epochs in range(epochs):
		t1=time.time()
		for num_batch in xrange(n_train_batchs):
			min_batch_cost.append(train_nn(num_batch))
		t2=time.time()
		print 'The %d/%dth training,takes %f seconds,cost is %f' %(num_epochs+1,epochs,(t2-t1),np.mean(min_batch_cost))
	return mynn	




if __name__ == '__main__':
    train_data,train_label = load_train('../data/train.csv')
    y_pre=np.argmax(train_label,1)
#	testdata = load_test('test.csv')
#	train_data = np.random.uniform(0,1,(100,20))
#	train_label = np.random.uniform(0,1,(100,2))
#	train_data,train_label = load_train('train.csv')
#	test_data = load_test('test.csv')
    testdata = T.matrix('../data/testdata')
#	testdata = theano.shared(np.asarray(testdata,dtype = np.float32),borrow=True)
    mynn = train(inputs=train_data,labels=train_label,n_in=784,n_out=10,learning_Rate=0.5,hidden_sizes=[500,500],batch_sizes=20,epochs=100)
    train_data = theano.shared(np.asarray(train_data,dtype = theano.config.floatX),borrow=True)	
#	test_data = theano.shared(np.asarray(test_data,dtype = theano.config.floatX),borrow=True)
    predict_nn = theano.function(inputs=[],outputs=mynn.predict_outputs(testdata),givens={testdata:train_data})
    predict = predict_nn()
    y = np.argmax(predict,1)
    error = abs(y-y_pre)
    acc = len(error[error==0])
    print acc
#	np.savetxt('output.csv',predict)
#	id = range(1,28001)
#	output = pd.DataFrame( data={"ImageId":id, "Label":y} )
#	output.to_csv( "predict.csv", index=False, quoting=3 )

