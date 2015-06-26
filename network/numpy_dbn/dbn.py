import numpy as np
import rbm as mymodel


class DBN(object):
	def __init__(self,sizes):
		self.sizes = sizes
		self.n = len(sizes)
		self.rbm = {}
		for i in range(1,self.n):
			subsizes=[self.sizes[i-1],self.sizes[i]]
			self.rbm[str(i)] = mymodel.RBM(subsizes[0],subsizes[1])
	def train(self,x):
		self.rbm[str('1')].train(x)
		for i in range(2,self.n):
			x = self.rbm[str(i-1)].up(x)
			self.rbm[str(i)].train(x)


if __name__ == '__main__':
	x=np.random.uniform(0,1,(500,50))
	mydbn=DBN([50,30,2])
	mydbn.train(x)
	print 'Great!'

