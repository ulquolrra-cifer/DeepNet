import numpy as np
import scipy as cp
import time
import nn2
from active_function import *

class Cnnlayers(object):
    def __init__(self,inputs,filter_shape,image_shape,poolsize=(2.0,2.0)):
		self.inputs = inputs
		fan_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
		fan_in = np.prod(filter_shape[1:])
		w_bound = np.sqrt(6.0/(fan_in+fan_out))
		self.W = np.random.uniform(-0.5,0.5,np.shape(filter_shape))*w_bound
		self.b = np.zeros((filter_shape[0],1))
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.filter_nums = filter_shape[0]
		self.image_nums = image_shape[0]
		self.channels = image_shape[1]
	def convolution(self):
		image_row = self.image_shape[2]
		image_col = self.image_shape[3]
		filter_row = self.filter_shape[2]
		filter_row = self.filter_shape[3]
		self.convolved_features = zeros((self.image_nums,self.filter_nums,(image_row-filter_row+1),(image_col-filter_col+1)))
		for i in range(self.image_nums):
			for j in range(self.filter_nums):
				convolved_images = np.zeros(((image_row_filter_row+1),(image_col-filter_col+1)))
				x = np.zeros(np.shape(convolved_images))
				for k in range(self.channels):
					im = self.inputs[i,k,:,:]
					w = self.filter_shape[j,k,:,:]
					convolved_images = cp.ndimage.convolve(im,w,'valid')
					x += convolved_images
				self.convolved_features[i,j,:,:] = x


	def mean_pool(self):
		pool_num=poolsize[0]
		result_dim=np.shape(self.convolved_features)[2] / pool_num
		pooled_features = zeros((self.image_nums,self.filter_nums,result_dim.result_dim))
		for i in range(self.image_nums):
			for j in range(self.filter_nums):
				
				for rol_nums in range(result_dim):
					offsetrow = rol_nums*pool_num
					for col_nums in range(result_dim):
						offsetcol = col_nums*pool_num
						patch = self.convolved_features[i,j,offsetrow:(offsetrow+pool_num),offsetcol:(offsetcol+pool_num)]
						self.pooled_features[i,j,rol_nums,col_nums] = np.mean(patch)
	def cnnoutputs(self):
		for i in range(self.image_nums):
			for j in range(self.filter_nums):
				self.pooled_features[i,j,:,:] = tanh_opt(self.pooled_features[i,j,:,:])
				

def ConvolutionNetworks(learning_rate,data_set):
	layer1 = Cnnlayers(inputs=data_set,image_shape=(batch_sizes,1,28,28),filter_shape=(6,1,5,5),poolsize=(2,2))
	layer2 = Cnnlayers(inputs=layer1.self.pooler_features,image_shape=(batch_sizes,6,12,12),filter_shape=(12,6,5,5),poolsize=(2,2))
	num1 = np.shape(layer2.pooled_features)[1]
	num2 = np.shape(layer2.pooled_features)[2]
	num3 = np.shape(layer2.pooled_features)[3]
	x = np.zeros((batch_sizes,num1*num2*num3))
	for i in range(batch_sizes):
		x[i,:] = layer2.pooled_features[i,:].flatten()
	mynn=nn2.NN([(num1*num2*num3),500,10]) 

































