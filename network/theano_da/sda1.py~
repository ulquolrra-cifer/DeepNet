import numpy as np
import theano.tensor as T
import theano
from my_layers import Hiddenlayer,OutputsLayer,form_label,unfold
from da import da
import time
from my_layers import load_data
from nn import ForwordNN
class Sda(object):
	def __init__(self,n_in,n_out,hidden_sizes,np_rng=None,theano_rng=None):
		self.da_layers = []
		self.hidden_layers = []
		self.n_layers = len(hidden_sizes)
		self.input = T.matrix('input')
		self.label = T.matrix('label')
		self.params = []
		for i in range(len(hidden_sizes)):
			if i == 0:
				inputs_size = n_in
			else:
				inputs_size = hidden_sizes[i-1]	
			
			if i == 0:
				layer_input = self.input
			else: 
				layer_input = self.hidden_layers[-1].output

			hidden_layer = Hiddenlayer(inputs=layer_input,n_in=inputs_size,n_out=hidden_sizes[i])
			self.hidden_layers.append(hidden_layer)
			self.params.extend(hidden_layer.params)
			da_layer = da(inputs=layer_input,n_v=inputs_size,n_h=hidden_sizes[i],w=hidden_layer.w,b_h=hidden_layer.b,rng=np_rng)
			self.da_layers.append(da_layer)
		self.output_layer = OutputsLayer(inputs=self.hidden_layers[-1].output,n_in=hidden_sizes[-1],n_out=n_out,labels=self.label)
		self.params.extend(self.output_layer.params)
		self.output = self.output_layer.output
		self.cost = self.output_layer.cost(self.output,self.label)


	def pretraining(self,inputs,batch_size,learning_rate,denoising):
		index = T.lscalar('index')
	#	batche_num = inputs.get_value(borrow=True).shape[0] / batch_size
		i=0
		pre = []
		for DA in self.da_layers:
			error,update = DA.get_cost_updates(denoising[i],learning_rate)
			i += 1
			train_da = theano.function(inputs=[index],
										outputs = error,
										updates = update,
										givens = {
											self.input:inputs[index*batch_size:(index+1)*batch_size]
											}
									)
			pre.append(train_da)
		return pre


			
			
	
