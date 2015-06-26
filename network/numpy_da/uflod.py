import numpy as np

def ufold(net1,net2):
	n = len(net1.W)
	for i in range(1,n+1):
		net2.W[str(i)] = net1.da[str(i)].W[str(i)]
