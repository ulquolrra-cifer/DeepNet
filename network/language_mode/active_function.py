import sys
import numpy as np
from scipy import special as sp_special

FLT_MAX = sys.float_info.max
def sigmoid(x): 
	return 1.0/(1.0+np.exp(-x))

def sigmoid_d(y):
	return y*(1.0-y)

def softmax(x):
	exp = np.exp(x)
	return exp / np.sum(exp)

def softmax_d(y):
	d_y = np.outer(-y,y)
	d_y[np.diag_indices_from(d_y)] += y 
	return d_y
