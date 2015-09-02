from __future__ import division
import numpy as np
import os
import time
import json
from collections import OrderedDict
from active_function import *
import data_process as dp
activation_hidden = sigmoid
activation_hidden_d = sigmoid_d
activation_output = softmax
activation_output_d = softmax_d

def init_matrix(rows,cols,low,high):
    matrix = (high-low) * np.random.random([rows,cols]) + low
    matrix += (high-low) * np.random.random([rows,cols]) + low
    matrix += (high-low) * np.random.random([rows,cols]) + low
    return matrix

def sparse_dot(words,vocabulary_size,context,hidden_size,w):
 #   print 'vocabulary_size is %d,hidden_size is %d' % (vocabulary_size,hidden_size)
  #  print 'context shape %d %d' % (context.shape[0],context.shape[1])
    results = np.dot(context,w[vocabulary_size:vocabulary_size+hidden_size,:])
    if (words == -1):
        return results
    else:
        return np.add(w[words,:],results)

class RecurrentNet(object):
    def __init__(self,vocabulary_name,train_size,train_name,valid_name,test_name,epoch,learnrate,hidden_size,init_matrix_min=-0.1,init_matrix_max=0.1):
        self.train = open(train_name,'r')
        self.valid = open(valid_name,'r')
        self.test = open(test_name,'r')
        self.epoch = epoch
        self.learnrate = learnrate
        self.hidden_size = hidden_size
        v = open(vocabulary_name,'r')
        vocabulary = json.load(v)
        v.close()
        ss_nums = vocabulary.pop('</s>')
        vocabulary = OrderedDict({'</s>':ss_nums}.items()+OrderedDict(sorted(vocabulary.items(),key=lambda x:(-x[1],x[0]))).items())
        self.vocabulary_nums = len(vocabulary)
        self.input_index = -1
        self.context = [0.1] * self.hidden_size
        self.hidden = np.zeros(self.hidden_size)
        self.output = np.zeros(self.vocabulary_nums)
        self.W_input = init_matrix(self.vocabulary_nums+self.hidden_size,self.hidden_size,init_matrix_min,init_matrix_max)
        self.W_hidden = init_matrix(self.hidden_size,self.vocabulary_nums,init_matrix_min,init_matrix_max)
    def reset_context(self):
        self.hidden = np.zeros(self.hidden_size)
        self.context = [0.1] * self.hidden_size
    def nnff(self,previous_word):

        self.hidden = activation_hidden(sparse_dot(previous_word,self.vocabulary_nums,self.context,self.hidden_size,self.W_input))
        self.output = activation_output(np.dot(self.hidden,self.W_hidden))

    def nnbp(self,previous_word,current_word):
        if (current_word == -1):
            return
        output_error = self.output
        output_error[current_word] -= 1
        output_deltas = output_error

        hidden_error = np.dot(self.W_hidden,output_deltas)
        hidden_deltas = activation_hidden_d(np.copy(self.hidden)) * hidden_error
        self.W_hidden -= self.learnrate*np.outer(self.hidden,output_deltas)

        if (previous_word != -1):
            self.W_input[previous_word,:] -= hidden_deltas*self.learnrate
        self.W_input[self.vocabulary_nums:,:] -= self.learnrate*np.outer(self.context,hidden_deltas)
        

    def nntrain(self):
        for epoch in xrange(1,self.epoch+1):
            previous_word = 0
            self.reset_context()
            self.train.seek(0)
            for word in self.train:
                word = int(word)
                self.nnff(previous_word)
                self.nnbp(previous_word,word)
                self.context = np.copy(self.hidden)
                previous_word = word
                if (word == 0):
                    self.reset_context()
            self.reset_context()
            previous_word = 0
            logp = 0
            nums = 0
            self.valid.seek(0)
            for word in self.valid:
                word = int(word)
                self.nnff(previous_word)
                if (word != -1):
                    nums += 1
                    logp += np.log2(self.output[word])
                self.context = np.copy(self.hidden)
                previous_word = word
                if (word == 0):
                    self.reset_context()
            print 'num %d' % (nums)
            print("****************************")
            print('{}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
            print('Epoch {}'.format(epoch))
            print('valid dataset log probability {}'.format(logp))
            print('valid dataset words nums {}'.format(nums))
            print('valid dataset PPL {}'.format(2.0**(-logp/nums)))
    def get_predict_word(self):
        return self.output.argmax()

    def nntest(self):
        self.reset_context()
        previous_word = 0
        logp = 0
        nums = 0
        error = 0
        for word in self.test:
            word = int(word)
            self.nnff(previous_word)
            if (word != -1):
                nums += 1
                logp += np.log2(self.output[word])
                if (word != 0 and word != self.get_predict_word):
                    error += 1
            self.context = np.copy(self.hidden)
            previous_word = word
            if (word == 0):
                self.reset_context()
        print('Test log probability {}'.format(logp))
        print('Test PPL {}'.format(2.0**(-logp/nums)))
        print('Test error rate {}'.format(error/nums))
