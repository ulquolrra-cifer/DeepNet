#use w and b to training the network 
import numpy as np
from active_function import *
import time


class NN(object): 
	def __init__(self,architecture,activation_function='sigm',learningRate = 1,momentum = 0.5,weightPenaltyL2 = 0,sparsityparameter = 0,beta=0,inputzeroMaskedFraction=0,output = 'sigm',jacobi_penalty = 0,scaling_learningRate = 0.99,dropout = 0):
		self.size=architecture
		self.n=len(self.size)
		self.activation_function = activation_function
		self.learningRate = learningRate
		self.momentum = momentum
		self.weightPenaltyL2 = weightPenaltyL2
		self.sparsityparameter = sparsityparameter
		self.beta = beta      #weigth of the sparsity penalty term
		self.inputzeroMaskedFraction = inputzeroMaskedFraction
#		self.droputFraction = 0
		self.testing = 0
		self.output = output
		self.W = {}
		self.b = {}
		self.P = {}
		self.a = {}
		self.dW = {}
		self.db = {}
		self.vW = {}
		self.dropout = dropout
#		self.my_params = my_params
		self.d = {}
#		self.d_pre = {}
		self.dropMask = {}
		self.jacobi_penalty = jacobi_penalty
		self.scaling_learningRate = scaling_learningRate
#		self.e
#		self.L
		numpy_rng = np.random.RandomState(1234)
#		self.D = np.ones((1,self.size[1]))/float(self.size[1])
		for i in range(1,self.n):
			self.W[str(i)] = (np.array(numpy_rng.uniform(low=0,high=1,size=(self.size[i-1],self.size[i])))-0.5)*2*4*np.sqrt(6.0/(self.size[i]+self.size[i-1]))
			self.b[str(i)] = (np.zeros((1,self.size[i])))	
			self.P[str(i)] = np.zeros((1,self.size[i]))
			self.vW[str(i)] = np.zeros(np.shape(self.W[str(i)]))
		
	def nnff(self,x,y):
		n = self.n
		row,col = np.shape(x)
		self.a['1'] = x.copy()
		self.P['1'] = np.sum(self.a['1'],0)
		for i in range(2,n):
			
			if self.activation_function == 'sigm':
				self.a[str(i)] = sigm(np.dot(self.a[str(i-1)],self.W[str(i-1)])+np.tile(self.b[str(i-1)],(self.a[str(i-1)].shape[0],1)))
			elif self.activation_function == 'tanh_opt':
				self.a[str(i)] = tanh_opt(np.dot(self.a[str(i-1)],self.W[str(i-1)])+np.tile(self.b[str(i-1)],(self.a[str(i-1)].shape[0],1)))

			if self.dropout > 0:
				self.dropMask[str(i)] = np.random.uniform(0,1,np.shape(self.a[str(i)]))>self.dropout
				if self.testing ==1:
					self.a[str(i)] = self.a[str(i)] * (1-self.dropout)
				else:
					self.a[str(i)] *= self.dropMask[str(i)]
#			if self.my_params > 0:
#				binarys = np.random.uniform(0,1,(self.a[str(i)].shape))>self.my_params
#				self.nums = np.sum(binarys,0)
#				a = self.a[str(i)]*binarys
#				self.a[str(i)] = a
			if self.sparsityparameter > 0:
				self.P[str(i)] = 0.01*np.mean(self.a[str(i)],0) + 0.99*self.p[str(i)]
		if self.output == 'sigm':
			self.a[str(n)] = sigm(np.dot(self.a[str(n-1)],self.W[str(n-1)])+np.tile(self.b[str(n-1)],(self.a[str(n-1)].shape[0],1)))
		elif self.output == 'linear':
			self.a[str(n)] = np.dot(self.a[str(n-1)],self.W[str(n-1)])+np.tile(self.b[str(n-1)],(self.a[str(n-1)].shape[0],1))
		elif self.output == 'softmax':
			self.a[str(n)] = softmax(np.dot(self.a[str(n-1)],self.W[str(n-1)])+np.tile(self.b[str(n-1)],(self.a[str(n-1)].shape[0],1)))		
		
		self.e = y - self.a[str(n)]	
		

		if self.output == 'sigm' or self.output == 'linear':
			self.L = 1.0/2.0*(np.sum(self.e**2))/row
		elif self.output == 'softmax':
			self.L = -np.sum(y * np.log(self.a[str(n)]))/row
	def nnbp(self):
		n=self.n
		d={}
		if self.output == 'sigm':
			self.d[str(n)] = -self.e*(self.a[str(n)]*(1-self.a[str(n)]))
		elif (self.output == 'softmax' or self.output == 'linear'):
			self.d[str(n)] = - self.e
		for i in range(n-1,1,-1):
			if self.activation_function == 'sigm':
				d_act = self.a[str(i)]*(1-self.a[str(i)])
			elif self.activation_function == 'tanh_opt':
				d_act = 1.7159*2.0/3.0*(1-1/(1.7159)**2 * self.a[str(i)]**2)
			
			if self.sparsityparameter == 0:
#
#				if i+1 == n:
				self.d[str(i)] = np.dot(self.d[str(i+1)],self.W[str(i)].T)*d_act
#				else:	
#					d[str(i)] = np.dot(d[str(i+1)][:,1:],self.W[str(i)])*d_act
			else:
				term = -(self.sparsityparameter / self.P[str(i)]) + (1-self.sparsityparameter) / (1.0 - self.P[str(i)])
				self.d[str(i)] = (np.dot(self.d[str(i+1)],self.W[str(i)].T)+self.beta*term)*d_act
#				term = -(self.sparsityparameter / self.P[str(i)]) + (1-self.sparsityparameter) / (1 - self.P[str(i)])
#				if i+1 == n:
#					d[str(i)] = (np.dot(d[str(i+1)],self.W[str(i)])+self.beta*term)*d_act
#				else:
#					d[str(i)] = (np.dot(d[str(i+1)][:,1:],self.W[str(i)])+self.beta*term)*d_act
			if self.dropout > 0:
				self.d[str(i)] *= self.dropMask[str(i)]  
		for i in range(1,n,1):
#			if i+1 == n:
			self.dW[str(i)] = np.dot(self.a[str(i)].T,self.d[str(i+1)])/np.shape(self.d[str(i+1)])[0]
			self.db[str(i)] = (self.d[str(i+1)].mean(0))[None,:]
#			else:
#				self.dW[str(i)] = np.dot(d[str(i+1)][:,1:].T,self.a[str(i)])/np.shape(d[str(i+1)])[0]
#		self.D = self.D * np.e**(self.nums*np.mean(abs(self.d[str(2)])-abs(self.d_pre[str(2)]),0))	
#		self.D = self.D / np.sum(self.D)
#		for i in range(len(self.d)):
#			self.d_pre[str(i+2)] = self.d[str(i+2)]

	def nnapplygrads(self,x):
		for i in range(self.n-1,0,-1):
			dw = self.dW[str(i)]
			db = self.db[str(i)]

			if self.jacobi_penalty > 0:
				if i==1:
					a = (self.a[str(i+1)]*(1-self.a[str(i+1)]))**2
					d = ((1-2*self.a[str(i+1)])*a*(self.W[str(i)]**2).sum(0)[None,:])
					b = np.dot(x.T/x.shape[0],d)
					c = a.mean(0)*self.W[str(i)]
					dw_jacobi = b+c
					db_jacobi = d.mean(0)
					dw = dw + self.jacobi_penalty * dw_jacobi
					db = db + self.jacobi_penalty * db	
			if self.weightPenaltyL2 > 0:
				dw = dw + self.weightPenaltyL2 * (self.W[str(i)])		
			dw = self.learningRate*dw
			db = self.learningRate*db
#			if self.jacobi_penalty > 0:
			if (self.momentum > 0):
				self.vW[str(i)] = self.momentum*self.vW[str(i)]+dw
				dw = self.vW[str(i)]
			self.W[str(i)] = self.W[str(i)]-dw
			self.b[str(i)] = self.b[str(i)]-db
	def nntrain(self,x,y,batchsize=1,numepochs=10):
		m = np.shape(x)[0]
		numbatches = m / float(batchsize)
	#	L = np.zeros((numepochs * numbatches,1))
		n = 1
#		for i in range(2,self.n+1):
#			self.d_pre[str(i)] = np.zeros((batchsize,self.size[i-1]))
		for i in range(numepochs):
			kk=range(m)
			np.random.shuffle(kk)
			kk = np.array(kk)
			t1=time.time()
			for l in range(int(numbatches)):
				batch_x = x[kk[(l)*batchsize : ((l+1) * batchsize)],:]
				if self.inputzeroMaskedFraction > 0:
					batch_x = batch_x * (np.random.uniform(0,1,(np.shape(batch_x))) > self.inputzeroMaskedFraction)
#					batch_x[np.random.uniform(0,1,(np.shape(batch_x)))<self.inputzeroMaskedFraction] = 0 
				batch_y = y[kk[(l)*batchsize : ((l+1) * batchsize)],:]
				self.nnff(batch_x,batch_y)
				self.nnbp()
				self.nnapplygrads(batch_x)
			self.learningRate = self.learningRate * self.scaling_learningRate
			t2=time.time()
			print '%d/%d th epochs takes %fseconds' % (i,numepochs,(t2-t1))
#				print l
	
	def nnpredict(self,x):
		self.testing = 1
		self.nnff(x,np.zeros((np.shape(x)[0],self.size[-1])))
		self.testing = 0
		label = self.a[str(len(self.size))]
		m = np.shape(x)[0]
		y = np.zeros((m,self.size[-1]))
		for i in range(m):
			tmp = np.argmax(label[i,:])
			y[i,tmp] = 1		
		return y
if __name__ == "__main__":
		x = np.random.uniform(0,1,(100,20))
		y = np.random.uniform(0,1,(100,2))
		mynn = NN([20,15,10,2])
		mynn.nnff(x,y)
		mynn.nnbp()
		mynn.nnapplygrads(x)
		label=mynn.nnpredict(x)
		print label













					



		

