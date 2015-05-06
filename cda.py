#This is the implement for contractive auto-encoder
import numpy 
import nn
class Cae(object):
	def _init_(self,architecture):
		self.size = architecture
		self.n = len(architecture)
		
