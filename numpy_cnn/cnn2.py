import numpy as np
import scipy assp
from active_fuction import *

class conlayer(object):
	def __init__(self,outputmaps,kernelsize):
		self.types = 'c'
		self.outputmaps = outputmaps
		self.kernelsize = kernelsize
		self.w = np.random.uniform(-0.5,0.5,(kernelsize,kernelsize))
		self.b = np.zeros((outputmaps,1))
class downsamplelayer(object):
	def __init__(self,scale):
		self.types = 's'
		self.scale = scale		
		

class Cnn(object):
	def __init__(self,structures,x,y):
		inputmaps = 1
		layers = len(structures)
		mapsize = np.shape(x)[2]
		for i in range(layers):
			if structures[i].types == 'c':
				mapsize = mapsize - sturcutres[i].outputmaps +1
				fan_out = structures[i].outputmaps * structures[i].kernelsize**2
				for j in range(structures[i].outputmaps):
					fan_in = inputmaps * structures[i].kernelsize**2
					for k in range(inputmaps):
						structures[i].w[k][j] 
