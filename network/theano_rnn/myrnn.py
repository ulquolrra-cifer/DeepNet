"""
Vanilla Recurrent Neural Network
Thank to gwtaylor for his codes available at: 
https://github.com/gwtaylor/theano-rnn

"""
import numpy as np
import theano.tensor as T
import theano
import time
from collections import OrderedDict
class RNN(object):
    def __init__(self,n_in,hidden,n_out,activation,output_type,error_type,learn_rate,moment,L1_reg,L2_reg):
        self.n_in = n_in
        self.hidden = hidden
        self.n_out = n_out
        self.activation = activation
        self.output_type = output_type
        self.error_type = error_type
        self.learn_rate = learn_rate
        self.moment = moment
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.w_ih = theano.shared(np.asarray(np.random.uniform(
                                                    size = (n_in, hidden),
                                                    low = -.01, high = .01),
                                                dtype = theano.config.floatX),
                                  name = 'w_ih')

        self.w_hh = theano.shared(np.asarray(np.random.uniform(
                                                        size = (hidden, hidden),
                                                        low = -.01, high = .01),
                                                    dtype = theano.config.floatX),
                                  name = 'w_hh')

        self.w_ho = theano.shared(np.asarray(np.random.uniform(
                                                    size = (hidden, n_out),
                                                    low = -.01, high = .01),
                                                dtype = theano.config.floatX),
                                  name = 'w_ho')

        # initial value of hidden layer units are set to zero
        self.h0 = theano.shared(np.zeros((hidden, ),
                                            dtype = theano.config.floatX),
                                name = 'h0')

        # biases are initialized to zeros
        self.b_h = theano.shared(np.zeros((hidden, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_h')

        self.b_o = theano.shared(np.zeros((n_out, ),
                                            dtype = theano.config.floatX),
                                 name = 'b_o')

        self.params = [self.w_ih, self.w_hh, self.w_ho, self.h0,
                       self.b_h, self.b_o]        
        self.updates = OrderedDict()
        for param in self.params:
            self.updates[param] = theano.shared(value = np.zeros(
                                                        param.get_value(
                                                        borrow = True).shape,
                                                        dtype = theano.config.floatX),
                                                name = 'updates')
        def step(u_t,h_tm1):
            if self.activation == 'sigmoid':
                h_t = T.nnet.sigmoid(T.dot(u_t,self.w_ih)+T.dot(h_tm1,self.w_hh)+self.b_h)
                y_t = T.dot(h_t,self.w_ho) + self.b_o
            elif self.activation == 'tanh':
                h_t = T.tanh(T.dot(u_t,self.w_ih)+T.dot(h_tm1,self.w_hh)+self.b_h)
                y_t = T.dot(h_t,self.w_ho) + self.b_o
            elif self.activation == 'relu':
                h_t = T.maximun((T.dot(u_t,self.w_ih)+T.dot(h_tm1,self.w_hh)+self.b_h),0)
                y_t = T.dot(h_t,self.w_ho) + self.b_o
            else:
                raise NotImplementedError 
            return h_t,y_t
        [self.h,self.y_pred], _ = theano.scan(step,self.x,[self.h0,None])
        self.L1 = abs(self.w_ih.sum()) + abs(self.w_hh.sum()) + abs(self.w_ho.sum())
        self.L2 = ((self.w_ih) **2 ).sum() + ((self.w_hh) **2 ).sum() + ((self.w_ho) **2 ).sum()
        if self.output_type == 'softmax':
            self.predicts = T.nnet.softmax(self.y_pred)
        elif self.output_type == 'sigmoid':
            self.predicts = T.nnet.sigmoid(self.y_pred)
        else:
            self.predicts = self.y_pred


        if self.error_type == 'mse':
            self.cost = T.mean((self.predicts - self.y) ** 2)
        elif self.error_type == 'crossentropy':
            self.cost = T.mean(T.nnet.binary_crossentropy(self.predicts, y))
        elif self.error_type == 'negativeloglihoo':
            self.cost = -T.mean(T.log(self.predicts)[T.arange(y.shape[0]), y])
        else:
                raise NotImplementedError 

    def train(self,train_x,train_y,test_x = None,test_y = None,epochs = 10):
        train_x = theano.shared(np.asarray(train_x, dtype=theano.config.floatX))
        train_y = theano.shared(np.asarray(train_y, dtype=theano.config.floatX))
        index = T.lscalar('index')
        cost = self.cost + self.L1 * self.L1_reg + self.L2 * self.L2_reg
        gparams =[]
        for param in self.params:
            gparam=T.grad(cost,param)
            gparams.append(gparam)
        updates = OrderedDict()
        for param,gparam in zip(self.params,gparams):
            weight_update = self.updates[param]
            upd = self.moment*weight_update-self.learn_rate*gparam
            updates[param] = upd
            updates[param] = param + upd
        train_rnn = theano.function([index],
                    cost,
                    updates = updates,
                    givens = {
                                self.x:train_x[index],
                                self.y:train_y[index]
                            }
                    )
        epoch = 0
        n_train = train_x.get_value(borrow = True).shape[0]
        while (epoch < epochs):
            errors = []
            t1 = time.time()
            for idx in xrange(n_train):
                errors.append(train_rnn(idx))
            t2 = time.time()
            print '%d/%d epoch,takes %f secondes,errors is %f' % (epoch,epochs,(t2-t1),np.mean(errors))
            epoch += 1  




def example():
    n_in = 3 # input vector size (not time at this point)
    hidden = 10 # hidden vector size
    n_out = 3 # output vector size
    time_steps = 15 # number of time-steps in time
    n_seq = 100 # number of sequences for training

    np.random.seed(0)
    
    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_in)
    targets = np.zeros((n_seq, time_steps, n_out))

    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 2:, 1] = seq[:, :-2, 1] # 2 time-step delay
    targets[:, 3:, 2] = seq[:, :-3, 2] # 3 time-step delay

    targets += 0.01 * np.random.standard_normal(targets.shape)

    myrnn = RNN(n_in = n_in, hidden = hidden, n_out=n_out,activation = 'tanh', output_type = 'real',error_type = 'mse',learn_rate = 0.001, L1_reg = 0, L2_reg = 0, moment = 0.9)
    myrnn.train(seq, targets) 


if __name__ == '__main__':
    example()















