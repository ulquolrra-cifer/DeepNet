
import numpy as np
import time
def sigm(x):
    return 1 / (1+np.exp(-x))
def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T
def tanh_opt(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class NN(object): 
    def __init__(self,architecture,activation_function='sigm',output = 'sigm'):
        self.size=architecture
        self.n=len(self.size)
        self.activation_function = activation_function
        self.output = output
        self.W = {}
        self.b = {}
        self.a = {}
        self.dW = {}
        self.db = {}
        self.d = {}
        self.d_act = {}
#        self.W,self.b = self.init_weights(1.0)
        psize = 0
        for i in xrange(len(architecture)-1):
            psize += architecture[i]*architecture[i+1]
        psize += np.sum(architecture[1:])
        self.psize = psize
        self.wsize = {}
        for i in range(1,self.n):
            self.wsize[i]=(i,self.size[i-1],self.size[i])
        self.W,self.b = self.init_weights(1.0)
        offsets = {}
        for i,j in zip(self.W,self.b):
            size_w_x,size_w_y = self.W[i].shape
            size_b = self.b[j].shape[1]
            offsets[i] = (i,size_w_x*size_w_y,size_b)
        self.offsets = offsets
    def init_weights(self,params):
        num_conn = 15
        W = {}
        b = {}
        for i in range(1,self.n):
            num,row,col = self.wsize[i]
            for j in range(col):
                W[i] = np.zeros((row,col))
                indices = np.random.choice(range(row-1),min(num_conn,row-1))
                W[i][indices,j] = np.random.randn(indices.size)*params
            b[i] = np.zeros((1,col))
            if self.activation_function == 'tanh':
                b[i] += 0.5
        return W,b
    def update_weights(self,dW,db):
        for i in range(1,self.n):
            self.W[i] += dW[i]
            self.b[i] += db[i]
    def J_dot(self, J, vec):

        if J.ndim == 2:
            return J * vec
        else:
            return np.einsum("ijk,ik->ij", J, vec)
    def compute_weights(self,vectors):
        dw,db = self.unpack(vectors)
        w = self.W
        b = self.b
        for i in range(1,self.n):
            w[i] += dw[i]
            b[i] += db[i]
        return w,b
    def get_error(self,w=None,b=None,inputs=None,targets=None):
        n = self.n
        if w is None:
            w = self.w
        else: 
            w = w
        if b is None:
            b = self.b
        else:
            b = b
        if inputs is None:
            inputs = self.inputs
        else:
            inputs = inputs
        if targets is None:
            targets = self.targets
        else:
            targets = targets
        a = {}
        a[1] = inputs
        for i in range(2,self.n):
            if self.activation_function == 'sigm':
                a[i] = sigm(np.dot(a[i-1],w[i-1])+np.tile(b[i-1],(a[i-1].shape[0],1)))
            elif self.activation_function == 'tanh_opt':
                a[i] = tanh_opt(np.dot(a[i-1],w[i-1])+np.tile(b[i-1],(a[i-1].shape[0],1)))
        if self.output == 'sigm':
            a[n] = sigm(np.dot(a[n-1],w[n-1])+np.tile(b[n-1],(a[n-1].shape[0],1)))
        elif self.output == 'linear':
            a[n] = np.dot(a[n-1],self.w[n-1])+np.tile(b[n-1],(a[n-1].shape[0],1))
        elif self.output == 'softmax':
            a[n] = softmax(np.dot(a[n-1],w[n-1])+np.tile(b[n-1],(a[n-1].shape[0],1)))
        error = np.sum((targets-a[n])**2)
        error /= 2*len(inputs)
        return error	

    def vec(self,x):
        return x.flatten()
    def pack(self,w,b):
        vectors = np.zeros((self.psize,))
        n_w = len(w)
        n_b = len(b)
        offsets = 0
        assert(n_w==n_b)
        for i,j in zip(w,b):
           # print i
            size_w_x,size_w_y = w[i].shape
            size_b = b[j].shape[1]
            vectors[offsets:(offsets+(size_w_x*size_w_y))] = self.vec(w[i])
            offsets += size_w_x*size_w_y
            vectors[offsets:(offsets+size_b)] = self.vec(b[j])
            offsets += size_b
        return vectors
    def unpack(self,vectors):
        w = {}
        b = {}
        offsets = 0
        for i in range(1,len(self.offsets)+1):
            num,size_w,size_b = self.offsets[i]
            _,m,n = self.wsize[i]
            w[i] = vectors[offsets:(offsets+size_w)].reshape((m,n))
            offsets += size_w
            b[i] = vectors[offsets:(offsets+size_b)].reshape((1,n))
            offsets += size_b
        return w,b
    def conjugate_gradient(self,init_delta,grad,iters=1):
        vals = {}
        store_iter = 5
        store_step = 1.3
        base_grad = -grad
        delta = init_delta
        deltas = {}
        residual = base_grad-self.compute_Gv(init_delta,self.damping)
        direction = residual.copy()
        res_norm = np.dot(residual,residual)
        temp_iters = 0
        for i in range(iters):
            G_dir = self.compute_Gv(direction,self.damping)
            alpha = res_norm / np.dot(direction,G_dir)
            delta += alpha*direction
            residual -= alpha * G_dir
            new_res_norm = np.dot(residual,residual)
  #          if new_res_norm < 1e-20:
   #             break
            beta = new_res_norm / res_norm
            direction = direction*beta + residual
            res_norm = new_res_norm
            if i == store_iter:
                deltas[i] = delta
                temp_iters = i
                store_iter = int(store_iter * store_step)
            vals[i] = -0.5 * np.dot(residual + base_grad,delta)
            k = max(int(0.1 * i),10)
            if (i>k and vals[i] < 0 and (vals[i]-vals[i-k])/vals[i] < k * 0.0005):
                break
        return temp_iters,deltas
    def compute_Gv(self,v,damping=0):
        Gv = np.zeros((self.psize,))
        Ra = {}
        Gvw = {}
        Gvb = {}
        RDa = {}
        RDs = {}
        #forward
        Ra[1] = np.zeros_like(self.a[1])
        Vw,Vb = self.unpack(v)
        for i in range(2,self.n+1):
            vw = Vw[i-1]
            vb = Vb[i-1]
            ww = self.W[i-1]
            R_input = np.zeros_like(self.a[i])
            R_input += np.dot(self.a[i-1],vw) + vb
            R_input += np.dot(Ra[i-1],ww)
 #           print '%d st R_input shape is %d,%d' % (i,R_input.shape[0],R_input.shape[1])
            Ra[i] = self.J_dot(self.d_act[i],R_input)
 #           print '%d st Ra shape is %d,%d' % (i,Ra[i].shape[0],Ra[i].shape[1])
        #backward
        RDa[self.n] = Ra[self.n]
        for i in range(self.n,1,-1):
            RDs[i]=self.J_dot(RDa[i],self.d_act[i])
 #           print '%d st RDs shape is %d,%d' % (i,RDs[i].shape[0],RDs[i].shape[1])
            Gvw[i-1] = np.dot(self.a[i-1].T,RDs[i])
            Gvb[i-1] = np.sum(RDs[i],0)[None,:]
  #          print 'W2 shape is %d %d' % (self.W[i-1].shape[0],self.W[i-1].shape[1])
            RDa[i-1] = np.dot(RDs[i],self.W[i-1].T)
  #       print 'W1 shape is %d %d' % (self.W[1].shape[0],self.W[1].shape[1])
  #       print 'W2 shape is %d %d' % (self.W[2].shape[0],self.W[2].shape[1])
  #       print 'Gvw1 shape is %d %d' % (Gvw[1].shape[0],Gvw[1].shape[1])
   #      print 'Gvw2 shape is %d %d' % (Gvw[2].shape[0],Gvw[2].shape[1])
  #       print 'b1 shape is %d %d' % (self.b[1].shape[0],self.b[1].shape[1])
   #      print 'b2 shape is %d %d' % (self.b[2].shape[0],self.b[2].shape[1])
  #       print 'Gvb1 shape is %d %d' % (Gvb[1].shape[0],Gvb[1].shape[1])
    #     print 'Gvb2 shape is %d %d' % (Gvb[2].shape[0],Gvb[2].shape[1])
        Gv += self.pack(Gvw,Gvb)
        Gv += damping * v
        return Gv
    def nnff(self,x,y):
        n = self.n
        row,col = np.shape(x)
        self.a[1] = x.copy()
        for i in range(2,n):
            if self.activation_function == 'sigm':
                self.a[i] = sigm(np.dot(self.a[i-1],self.W[i-1])+np.tile(self.b[i-1],(self.a[i-1].shape[0],1)))
            elif self.activation_function == 'tanh_opt':
                self.a[i] = tanh_opt(np.dot(self.a[i-1],self.W[i-1])+np.tile(self.b[i-1],(self.a[i-1].shape[0],1)))
        if self.output == 'sigm':
            self.a[n] = sigm(np.dot(self.a[n-1],self.W[n-1])+np.tile(self.b[n-1],(self.a[n-1].shape[0],1)))
        elif self.output == 'linear':
            self.a[n] = np.dot(self.a[n-1],self.W[n-1])+np.tile(self.b[n-1],(self.a[n-1].shape[0],1))
        elif self.output == 'softmax':
            self.a[n] = softmax(np.dot(self.a[n-1],self.W[n-1])+np.tile(self.b[n-1],(self.a[n-1].shape[0],1)))
        self.e = y - self.a[n]
		

        if self.output == 'sigm' or self.output == 'linear':
            self.L = 1.0/2.0*(np.sum(self.e**2))/row
        elif self.output == 'softmax':
            self.L = -np.sum(y * np.log(self.a[n]))/row
    def nnbp(self):
        n=self.n
        self.d_act = {}
        if self.output == 'sigm':
            self.d[n] = -self.e*(self.a[n]*(1-self.a[n]))
        elif (self.output == 'softmax' or self.output == 'linear'):
            self.d[n] = - self.e
        self.d_act[n] = self.d[n]
        for i in range(n-1,0,-1):
            if self.activation_function == 'sigm':
                self.d_act[i] = self.a[i]*(1-self.a[i])
            elif self.activation_function == 'tanh':
                self.d_act[i] = 1.7159*2.0/3.0*(1-1/(1.7159)**2 * self.a[i]**2)
            self.d[i] = np.dot(self.d[i+1],self.W[i].T)*self.d_act[i]
        for i in range(1,n,1):
            self.dW[i] = np.dot(self.a[i].T,self.d[i+1])/np.shape(self.d[i+1])[0]
            self.db[i] = (self.d[i+1].mean(0))[None,:]

    def nnapplygrads(self,x):
        for i in range(self.n-1,0,-1):
            dw = self.dW[i]
            db = self.db[i]		
            dw = self.learningRate*dw
            db = self.learningRate*db
            self.W[i] = self.W[i]-dw
            self.b[i] = self.b[i]-db
    def predict(self,testdata):
        self.nnff(testdata,np.zeros((testdata.shape[0],10)))
        predicts = self.a[self.n]
        
        return predicts
	
    def run_hf(self,inputs,targets,iters=100,init_damping=1.0,epochs=20,batch_size=None):
        init_delta = np.zeros((self.psize,))
        self.damping = init_damping

        for i in range(epochs):
            t1 = time.time()
            if batch_size is None:
                self.inputs = inputs
                self.targets = targets
            else:
                indices = np.random.choice(np.arange(len(inputs)),size=batch_size, replace=False)
                self.inputs = inputs[indices]
                self.targets = targets[indices]
            self.nnff(self.inputs,self.targets)
            self.nnbp()
            error = self.L
            grad = self.pack(self.dW,self.db)
  #          print "ready to CG"
            temp_iters,deltas = self.conjugate_gradient(init_delta*0.95,grad,iters)
            init_delta = deltas[temp_iters]
            new_err = np.inf
  #          print 'CG backtracking'
            #CG backtracking
            keys = deltas.keys()
            keys = sorted(keys,reverse=True)
            for j in keys:
                temp_w,temp_b = self.compute_weights(deltas[j])
                pre_err = self.get_error(temp_w,temp_b)
                if pre_err > new_err:
                    break
                delta = deltas[j]
                new_err = pre_err
            #damping
 #           print 'damping'
            denom = (0.5 * np.dot(delta,self.compute_Gv(delta,damping=0)) + np.dot(grad,delta))
            p = (new_err - error) / denom
            if p < 0.25:
                self.damping *= 1.5
            elif p > 0.75:
                self.damping *= 0.66
            #learn search
#            print 'learn_rate search'
            learn_rate = 1.0
            min_improv = min(1e-2 * np.dot(grad,delta),0)
            for _ in range(50):
                if new_err <= error + learn_rate * min_improv:
                    break
                learn_rate *= 0.8
                t_w,t_b = self.compute_weights(learn_rate * delta)
                new_err = self.get_error(t_w,t_b)
            else:
                learn_rate =0.0
                new_err = error
            dW,db = self.unpack(learn_rate * delta)
            self.update_weights(dW,db)
            t2 = time.time()
            print "%d/%d takes %f seconds the error is %f" % (i,epochs,(t2-t1),error)



if __name__ == '__main__':
    inputs = np.asarray([[0.1,0.1],[0.1,0.9],[0.9,0.1],[0.9,0.9]],dtype=np.float32)
    targets = np.asarray([[0.1],[0.9],[0.9],[0.1]],dtype=np.float32)
    ff = NN([2,5,1])
    ff.run_hf(inputs,targets)






					



		

