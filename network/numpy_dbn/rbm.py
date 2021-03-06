import numpy as np
from active_function import *

class RBM(object):
	def __init__(self,inputs,outputs,learning_rate = 1,moment = 0.5):
		self.inputs = inputs
		self.outputs = outputs
		self.learning_rate = learning_rate
		self.moment = moment
		self.W = np.random.uniform(-4.0*np.sqrt(6./(self.inputs+self.outputs)),4.0*np.sqrt(6./(self.inputs+self.outputs)),(self.inputs,self.outputs))
		self.b = np.zeros((self.inputs,1))
		self.c = np.zeros((self.outputs,1))
		self.vw = np.zeros((np.shape(self.W)))
		self.vb = np.zeros((np.shape(self.b)))
		self.vc = np.zeros((np.shape(self.c)))
	def up(self,x):
		return sigm(np.dot(x,self.W)+(self.c).T) 
	def down(self,x):
		return sigmrnd(np.dot(x,(self.W).T)+(self.b).T)
	def sample(self,visible):
		while(True):
			hidden = self.up(visible)
			yield visible,hidden
			visible = self.down(hidden)	
	def train(self,x,batchsize=1,numepochs=10):
		numbatches = np.shape(x)[0]/batchsize
		for i in range(numepochs):
			kk = range(np.shape(x)[0])
			np.random.shuffle(kk)
			kk = np.array(kk)
			for j in range(numbatches):
				batch_x = x[kk[j*batchsize:(j+1)*batchsize],:]
				item = self.sample(batch_x)
				v0,h0 = item.next()
				v1,h1 = item.next()
				self.vw = self.moment*self.vw + self.learning_rate * ((np.dot(v0.T, h0) - np.dot(v1.T, h1)) / np.shape(batch_x)[0])
				self.vb = self.moment*self.vb+ self.learning_rate*(v0 - v1).mean(axis=0)
				self.vh = self.moment*self.vc + self.learning_rate*(h0 - h1).mean(axis=0)
				self.W = self.W + self.vw
				self.b = self.b + self.vb
				self.c = self.c + self.vc
if __name__ == '__main__':
	x=np.random.uniform(0,1,(100,20))
#	y=np.random.uniform(0,1,(100,2))
	
	myrbm = RBM(20,2)
	myrbm.train(x)
	print 'Trian is done!'				
						  	
