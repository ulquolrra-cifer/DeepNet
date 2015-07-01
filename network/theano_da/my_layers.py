import numpy as np
import theano
import theano.tensor as T
import gzip
import cPickle
def form_label(label):
    num_label=label.get_value()
    m=label.get_value(borrow=True).shape[0]
    l=np.zeros((m,10))
    for i in range(m):
        if num_label[i]==0.:       				
            l[i][0]=1
        elif num_label[i]==1.0:
            l[i][1]=1
        elif num_label[i]==2:
            l[i][2]=1
        elif num_label[i]==3:
            l[i][3]=1
        elif num_label[i]==4:
            l[i][4]=1
        elif num_label[i]==5:
            l[i][5]=1
        elif num_label[i]==6:
            l[i][6]=1
        elif num_label[i]==7:
            l[i][7]=1
        elif num_label[i]==8:
            l[i][8]=1
        elif num_label[i]==9:
            l[i][9]=1
    return theano.shared(np.asarray(l,dtype=theano.config.floatX),borrow=True)
def unfold(sda,nn,dropout):
	n=sda.n_layers
	for i in range(n):
		nn.layer[i].w.set_value((sda.da_layers[i].w.get_value()*(1-dropout[i][0])))
		nn.layer[i].b.set_value(sda.da_layers[i].b_h.get_value())
	return nn
def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
        return shared_x,shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    return rval

	


class Hiddenlayer(object):
	def __init__(self,inputs,n_in,n_out,w=None,b=None,activation='sigmoid',rng=None):
		if rng is None:
			rng = np.random.RandomState(1234)
#		super(Hiddenlayer, self).__init__(input,n_in,n_out,activation = activation,rng = rng)
		self.input = inputs
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		if w is None:
			w = np.asarray(rng.uniform(
					low=-np.sqrt(6./(n_in+n_out)),
					high=np.sqrt(6./(n_in+n_out)),
					size=(self.n_in,self.n_out)
					),
					dtype=theano.config.floatX
				)
			if self.activation == 'sigmoid':
				w *= 4.0
			self.w = theano.shared(value=w,
						name='w',
						borrow=True)
		else:
			self.w = w
		if b is None:
			self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
		else:
			self.b = b
		self.params = [self.w,self.b]
		self.output=self.outputs(self.input)
	def outputs(self,inputs):
		if self.activation == 'sigmoid':
			outputs = T.nnet.sigmoid((T.dot(inputs,self.w)+self.b))
		elif self.activation == 'tanh':
			outputs = T.tanh((T.dot(inputs,self.w)+self.b))
		elif self.activation == 'linear':
			outputs = T.maximum(0,(T.dot(inputs,self.w)+self.b))
		return outputs
class OutputsLayer(object):
	def __init__(self,inputs,n_in,n_out,labels,w=None,b=None,activation='sigmoid',rng=None,costtype='meansquare'):
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		self.input = inputs
		self.label = labels
		self.costtype = costtype
		if w is None:
			w = np.zeros((n_in,n_out),dtype=theano.config.floatX)
			if self.activation == 'sigmoid':
				w *= 4.0
			self.w = theano.shared(value=w,
						name='w',
						borrow=True)
		else:
			self.w = w
		if b is None:
			self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
		else:
			self.b = b
		self.params = [self.w,self.b]
		self.output=self.outputs(self.input)
	def outputs(self,inputs):
		if self.activation == 'sigmoid':
			outputs = T.nnet.sigmoid(T.dot(inputs,self.w)+self.b)
		elif self.activation == 'softmax':
			outputs = T.nnet.softmax(T.dot(inputs,self.w)+self.b)

		elif self.activation == 'linear':		
			outputs = T.maximum((T.dot(inputs,self.w)+self.b),0)
		return outputs

	def cost(self,output,label):
#		self.outputs()
		if self.costtype == 'meansquare':
			return T.mean(((label- self.output)**2).sum(1))
		elif self.costtype == 'meanabsolute':
			return T.mean(abs(label-self.output).sum(1))
		elif self.costtype == 'negativeloglihood':
			return -T.mean(T.log(T.sum(self.output*label,1)))
		elif self.costtype == 'crossentropy':
			return -T.mean((label*T.log(self.output)+(1-label)*T.log(1-self.output)).sum(1))
#	-T.mean(self.targets*T.log(self.outputs) + (1-self.targets)*T.log(1-self.outputs))
#T.mean(((self.output - label)**2).sum(axis=1))
#-T.log(T.sum(self.outputs*self.targets))
#		elif self.activation == 'softmax':
#			return T.mean((T.nnet.binary_crossentropy(self.output, label)).mean(axis=1)) 
 
