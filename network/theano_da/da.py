import numpy as np
import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
#from my_layers import load_data
from my_layers import load_data
import time
from collections import OrderedDict
try:
    import PIL.Image as Image
except ImportError:
    import Image


class da(object):
    def __init__(self,inputs,n_v,n_h,rng,w = None,b_h = None,b_v = None,L1_reg=0.00,L2_reg=0.00,mom=0.5,activation = 'sigmoid',theano_rng = None,costtype='meansquare'):
        self.input = inputs
        self.activation = activation
        self.n_v = n_v
        self.n_h = n_h
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.mom = mom
        self.costtype = costtype
        if theano_rng is None:
            self.theano_rng = RandomStreams(rng.randint(2**30))
        if w is None:
            w = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6./(n_v+n_h)),
                    high=np.sqrt(6./(n_v+n_h)),
                    size=(self.n_v,self.n_h),
                    ),
                dtype = theano.config.floatX
            )
            if self.activation == 'sigmoid':
                w *= 4.0
            self.w = theano.shared(value=w,
                        name='w',
                        borrow=True)
        else:
            self.w = w			
        if b_h is None:
            self.b_h = theano.shared(value=np.zeros((n_h,),dtype=theano.config.floatX),name='b_h',borrow=True)
        else:
            self.b_h = b_h

        if b_v is None:
            self.b_v = theano.shared(value=np.zeros((n_v,),dtype=theano.config.floatX),name='b_v',borrow=True)
        else:
            self.b_v = b_v

        self.w_prime = self.w.T
        self.params = [self.w,self.b_h,self.b_v]
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)
#		self.L1 = 0.0
#		self.L2 = 0.0
        self.L1 = abs(self.w.sum())
        self.L2 = (self.w**2).sum()
    def v_to_h(self,inputs):
        if self.activation == 'sigmoid':
            return T.nnet.sigmoid(T.dot(inputs,self.w)+self.b_h)
        elif self.activation == 'tanh':
            return T.tanh(T.dot(inputs,self.w)+self.b_h)
        elif self.activation == 'linear':
            return T.maximum(0,T.dot(inputs,self.w)+self.b_h)
    def h_to_v(self,hiddens):
        if self.activation == 'sigmoid':
            return T.nnet.sigmoid(T.dot(hiddens,self.w.T)+self.b_v)
        elif self.activation == 'tanh':
            return T.tanh(T.dot(hiddens,self.w.T)+self.b_v)
        elif self.activation == 'linear':
            return T.maximum(0,T.dot(hiddens,self.w.T)+self.b_v)

    def denoising(self,inputs,de_params):
        return self.theano_rng.binomial(size = inputs.shape,n=1,p=1-de_params,dtype=theano.config.floatX)*inputs
#    def dropout(self,hiddens,drop_params):
#        return T.cast(self.theano_rng.binomial(size=hiddens.shape,n=1,p=1-drop_params),theano.config.floatX)*hiddens
    def get_cost_updates(self,learning_rate,params_denoising=0.3):
        if params_denoising > 0:
            denoising_inputs = self.denoising(self.input,params_denoising)
            y = self.v_to_h(denoising_inputs)
            z = self.h_to_v(y)
        else:
            y = self.v_to_h(self.input)
            z = self.h_to_v(y)
#		y = self.v_to_h(denoising_inputs)
#		z = self.h_to_v(hiddens)
        if self.costtype == 'meansquare':
            error = T.mean(((z- self.input)**2).sum(1))
        elif self.costtype == 'meanabsolute':
            error = T.mean(abs(self.input-z).sum(1))
        elif self.costtype == 'negativeloglihood':
            error = -T.mean(T.log(T.sum(z*self.input,1)))
        elif self.costtype == 'crossentropy':
            error = -T.mean((self.input*T.log(z)+(1-self.input)*T.log(1-z)).sum(1)) 
#		errors = T.mean(((z - self.input)**2).sum(1))
        cost = error+self.L1_reg*self.L1+self.L2_reg*self.L2
        gparams = []
        for param in self.params:
            gparam = T.grad(cost,param)
            gparams.append(gparam)
        updates = OrderedDict()
        for param,gparam in zip(self.params,gparams):
            weight_update = self.updates[param]
            upd = self.mom*weight_update - learning_rate*gparam
            updates[weight_update] = upd
            updates[param] = param + upd
#		dparams = T.grad(cost,self.params)
#		updates = [
#			(param,param - learning_rate*dparam)
#			for param,dparam in zip(self.params,dparams)
#		]
        return error,updates
def example(learning_rate,de_params,batchsizes,epochs,dataset='mnist.pkl.gz'):
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	n_batches = train_set_x.get_value(borrow=True).shape[0] / batchsizes
	np_rng = np.random.RandomState(89677)
	index = T.lscalar()
	x = T.matrix('x')
	myda = da(inputs=x,n_v=784,n_h=500,rng=np_rng)
	error,update = myda.get_cost_updates(de_params,learning_rate)
	train_da = theano.function([index],
								error,
								updates = update,
								givens = {
									x:train_set_x[index*batchsizes:(index+1)*batchsizes]
								}
							)
	mean_cost = []
	for n_batch in xrange(epochs):
		t1 = time.time()
		for n_index in xrange(n_batches):
			mean_cost.append(train_da(n_index))
		t2 = time.time()
		print '%d/%dth takes %f seconds and the mean cost is %f' % (n_batch,epochs,(t2-t1),np.mean(mean_cost))


if __name__ == '__main__':
	example(learning_rate=1,de_params=0.0,batchsizes=20,epochs=50)
		
