import numpy as np
import pickle
import dataproc
''' An implementation of an MLP with a single layer of hidden units. '''


class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.

        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.

        Note: a1 and z1 can be used for caching during backprop/evaluation.

        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units
        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2*(np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2*(np.random.random((self.dout, self.hidden_units)) - 0.5)

    def save(self, filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def eval(self, xdata):
        outputs = np.zeros(shape=(xdata.shape[1], self.dout))
        vs = xdata.transpose()
        for k in range(len(xdata[0])):
            a1 = np.zeros(shape=(self.hidden_units, 1))
            for i in range(self.hidden_units):
                a = self.activate(self.W1[i], vs[k]) + self.b1[i]
                a1[i] = np.tanh(a)
            for i in range(self.dout):
                a2 = (self.activate(self.W2[i], a1) + self.b2[i])[0]
                outputs[k] = np.tanh(a2)
        for output in outputs:
            output = np.max(outputs) - output
        for output in outputs:
            output = self.softmax(output)
        return outputs.transpose()
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''

    def sgd_step(self, xdata, ydata, learn_rate):
        # Calculate the squared error
        dEdW1, dEdb1, dEdW2, dEdb2 = self.grad(xdata, ydata)
        dEdW2.reshape((self.dout, self.hidden_units))
        # updating the weights by the learning rate times their gradient
        self.W1 -= np.multiply(dEdW1,learn_rate)
        self.W2 -= dEdW2*learn_rate
        
        # update the biases the same way
        self.b1 -= learn_rate * dEdb1
        self.b2 -= learn_rate * dEdb2
        ''' Do one step of SGD on xdata/ydata with given learning rate. '''

    def grad(self, xdata, ydata):
        outputs = np.zeros(shape=(xdata.shape[1], self.dout))
        vs = xdata.transpose()
        for k in range(len(xdata[0])):
            a1 = np.zeros(shape=(self.hidden_units, 1))
            for i in range(self.hidden_units):
                a = np.tanh(self.activate(self.W1[i], vs[k]) + self.b1[i])
                a1[i] = a
            for i in range(self.dout):
                a2 = np.tanh((self.activate(self.W2[i], a1) + self.b2[i])[0])

                outputs[k] = a2
        '''for output in outputs:
            output = np.max(outputs) - output
        for output in outputs:
            output = self.softmax(output)'''
        dEdW2 = np.zeros((self.dout, self.hidden_units)).transpose()

        dEdW1 = np.zeros((self.hidden_units, self.din))
        for k in range(len( vs)):
            E = outputs.transpose()[0][k] - ydata[0][k]
            error  = (E* self.tanh_der(np.arctan(a2)))
            for i in range(self.hidden_units):
               
                dEdW2[i] += error * a1[i]
                for j in range(self.din):
                    x = vs[k][j]
                    
                    w = error* self.W2.transpose()[i] * self.tanh_der(np.arctan(a1[i]))[0] * x
                    
                    dEdW1[i][j] += w[0]
        
        future = sum(sum(dEdW2))
        
        dEdb2 = np.zeros((1, 1))
        sums = 0
        E = outputs.transpose()[0] - ydata[0]
        for i in range(len(ydata[0])):
            sums += E[i]*np.arctanh(outputs[i])
        dEdb2[0][0] = sum(sums)
        dEdb1 = np.zeros((self.hidden_units, 1))
        for i in range(len(dEdb1)):
            dEdb1[i] = future * self.tanh_der(a1[i])[0]
        ''' Return a tuple of the gradients of error wrt each parameter.

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        return (dEdW1, dEdb1, dEdW2, dEdb2)

    def activate(self, weights, inputs):
        return np.dot(weights, inputs)

    def tanh_der(self, x):
        return 1 - np.tanh(x) ** 2

    def softmax(self, y):
        exp = np.exp(y)
        p = np.nansum(exp, axis=0).reshape((-1, 1))
        return exp/p

    def cross_entropy(self, outputs, targets):
        return sum(sum(targets * np.log(outputs)))

    def softmax_cross_entropy(self, outputs, targets):
        return - np.nansum(targets * np.log(softmax(outputs)), axis=0)


'''def main():
    xtrain, ytrain = dataproc.load_data("\\Users\\HUSSA\\Documents\\School\\CSC 246\\Projects\\Project 2\\data\\xorSmoke")
    N = xtrain.shape[1]
    din = xtrain.shape[0]
    dout = ytrain.shape[0]
    mlp = MLP(din, dout, 2)
    outputs = mlp.eval(xtrain)
    mlp.sgd_step(xtrain, ytrain, 0.1)
main()'''

