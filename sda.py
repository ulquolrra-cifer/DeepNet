import numpy as np
from nn2 import *

class Sda(object):
	def __init__(self,architecture):
		self.size = architecture
		self.n = len(architecture)
		self.da = {}
		for i in range(1,self.n):
			sub_architecture = [self.size[i-1],self.size[i],self.size[i-1]]
#			self.da[str(i)] = NN(sub_architecture,architecture,activation_function,learningRate,output,scaling_learningRate)
			self.da[str(i)] = NN(sub_architecture)
	def train(self,x,batchsize,numepochs):
		for i in range(1,self.n):
			self.da[str(i)].nntrain(x,x,batchsize,numepochs)
	#		print i
			self.da[str(i)].nnff(x,np.zeros((np.shape(x)[0],self.da[str(i)].size[-1])))
			t = self.da[str(i)].a['2']
			x = t.copy()
		
