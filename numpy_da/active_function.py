import numpy as np

def sigm(x):
	return 1.0 / (1+np.exp(-x))
def sigmrnd(x):
	return (1.0/(1+np.exp(-x))) > np.random.rand(*x.shape)

def softmax(x):
    e = np.exp(x - np.max(x))  
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  

def tanh_opt(x):
	return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
