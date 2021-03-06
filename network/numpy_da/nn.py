#use w to training network
import numpy as np
from active_function import *



class NN(object): 
	def __init__(self,architecture,activation_function='sigm',learningRate = 2,momentum = 0.5,weightPenaltyL2 = 0,sparsityparameter = 0,beta=0,sparsityparameter = 0,inputzeroMaskedFraction=0,output = 'sigm',jacobi_penalty = 0,scaling_learningRate = 0.99,hidelayer_number=100):
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
		self.P = {}
		self.a = {}
		self.dW = {}
		self.vW = {}
		self.jacobi_penalty = jacobi_penalty
		self.scaling_learningRate = scaling_learningRat
		self.D = np.ones((hidelayer_number,1))/float(hidelayer_number)
#		self.e
#		self.L
		numpy_rng = np.random.RandomState(1234)
		
		for i in range(1,self.n):
			self.W[str(i)] = (np.array(numpy_rng.uniform(low=0,high=1,size=(self.size[i],self.size[i-1]+1)))-0.5)*2*4*np.sqrt(6.0/(self.size[i]+self.size[i-1]))
			self.P[str(i)] = np.zeros((1,self.size[i]))
			self.vW[str(i)] = np.zeros(np.shape(self.W[str(i)]))
	def nnff(self,x,y):
		n = self.n
		row,col = np.shape(x)
		x_a = {}
		x_a['1'] = np.zeros((row,col+1))
		x_a['1'][:,0] = np.ones(row)
		x_a['1'][:,1:] = x.copy()
		self.a['1'] = x_a['1']
		for i in range(2,n):
			
			if self.activation_function == 'sigm':
				x_a[str(i)] = sigm(np.dot(self.a[str(i-1)],self.W[str(i-1)].T))
			elif self.activation_function == 'tanh_opt':
				x_a[str(i)] = tanh_opt(np.dot(self.a[str(i-1)],self.W[str(i-1)].T))

			
			row_a,col_a = np.shape(x_a[str(i)])
			self.a[str(i)] = np.zeros((row_a,col_a+1))
			self.a[str(i)][:,0] = np.ones(row_a)
			self.a[str(i)][:,1:] = x_a[str(i)]
			if self.sparsityparameter > 0:
				self.P[str(i)] = np.sum(self.a[str(i)],2)
		if self.output == 'sigm':
			self.a[str(n)] = sigm(np.dot(self.a[str(n-1)],self.W[str(n-1)].T))
		elif self.output == 'linear':
			self.a[str(n)] = np.dot(self.a[str(n-1)],self.W[str(n-1).T])
		elif self.output == 'softmax':
			self.a[str(n)] = softmax(np.dot(self.a[str(n-1)],self.W[str(n-1)].T))		
		
		self.e = y - self.a[str(n)]	
		

		if self.output == 'sigm' or self.output == 'linear':
			self.L = 1.0/2.0*(np.sum(self.e * 2))/row
		elif self.output == 'softmax':
			self.L = -np.sum(y * np.log(self.a[str(n)]))/row

	def nnbp(self):
		n=self.n
		d={}
		if self.output == 'sigm':
			d[str(n)] = -self.e*(self.a[str(n)]*(1-self.a[str(n)]))
		elif (self.output == 'softmax' or self.output == 'linear'):
			d[str(n)] = - self.e
		for i in range(n-1,1,-1):
			if self.activation_function == 'sigm':
				d_act = self.a[str(i)]*(1-self.a[str(i)])
			elif self.activation_function == 'tanh_opt':
				d_act = 1.7159*2.0/3.0*(1-1/1.7159)**2*self.a[str(i)]**2
			
			if self.sparsityparameter == 0:

				if i+1 == n:
					d[str(i)] = np.dot(d[str(i+1)],self.W[str(i)])*d_act
				else:
					d[str(i)] = np.dot(d[str(i+1)][:,1:],self.W[str(i)])*d_act
			else:
				term = -(self.sparsityparameter / self.P[str(i)]) + (1-self.sparsityparameter) / (1 - self.P[str(i)])
				if i+1 == n:
					d[str(i)] = (np.dot(d[str(i+1)],self.W[str(i)])+self.beta*term)*d_act
				else:
					d[str(i)] = (np.dot(d[str(i+1)][:,1:],self.W[str(i)])+self.beta*term)*d_act 
		for i in range(1,n,1):
			if i+1 == n:
				self.dW[str(i)] = np.dot(d[str(i+1)].T,self.a[str(i)])/np.shape(d[str(i+1)])[0]
			else:
				self.dW[str(i)] = np.dot(d[str(i+1)][:,1:].T,self.a[str(i)])/np.shape(d[str(i+1)])[0]
	def nnapplygrads(self):
		for i in range(self.n-1,0,-1):
			dw = self.dW[str(i)]
			dw = self.learningRate*dw
			if self.jacobi_penalty > 0:
				a = (self.a[str(i)]*(1-self.a[str(i)]))**2
				d = ((1-2*self.a[str(i)])*a*(self.W[str(i-1)]**2).sum(0)[None,:])					
			if self.momentum > 0:
				self.vW[str(i)] = self.momentum*self.vW[str(i)]+dw
				dw = self.vW[str(i)]
			self.W[str(i)] = self.W[str(i)]-dw
	def nntrain(self,x,y,batchsize,numepochs):
		m = np.shape(x)[0]
		numbatches = m / float(batchsize)
	#	L = np.zeros((numepochs * numbatches,1))
		n = 1
		
		for i in range(numepochs):
			kk=range(m)
			np.random.shuffle(kk)
			kk = np.array(kk)
			for l in range(int(numbatches)):
			
		
				batch_x = x[kk[(l)*batchsize : ((l+1) * batchsize)],:]
				if self.inputzeroMaskedFraction > 0:
					batch_x[np.random.uniform(0,1,(batchsize,np.shape(x)[1]))<self.inputzeroMaskedFraction] = 0 
				batch_y = y[kk[(l)*batchsize : ((l+1) * batchsize)],:]
				self.nnff(batch_x,batch_y)
				self.nnbp()
				self.nnapplygrads()
				self.learningRate = self.learningRate * self.scaling_learningRate

	
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














					



		

