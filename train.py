'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import os
import sys
import numpy as np

class mlp(object):
    def __init__(self, inpdim, hdim, outdim):
        self.w1 = np.random.randn(hdim, inpdim)
        self.b1 = np.random.randn(hdim, 1)
        self.w2 = np.random.randn(outdim, hdim)
        self.b2 = np.random.randn(outdim, 1)
        #print(self.w1, self.b1, self.w2, self.b2)

    def feedforward(self, x):
        a1 = x
        a2 = self.sigmoid(np.dot(self.w1, a1)+self.b1)
        a3 = self.sigmoid(np.dot(self.w2, a2)+self.b2)
        #print("Shape of a3",a3.shape)
        return a3
    
    def sgd(self, trainx, trainy):
        itr = 5     #number of iterations
        trainsize = len(trainx)     #number of training samples
        minbatchsize = trainsize//100
        for i in range(itr):
            print("training iteration: ", i)
            k = 0
            for j in range(0, trainsize, minbatchsize):
                self.updateweights(trainx[k:minbatchsize], trainy[k:minbatchsize])
                #print(k, k+minbatchsize)
                k = k + minbatchsize
            #print("b2: ", self.b2)
        np.savetxt("weights/w1.txt", self.w1)
        #Load this file using np.loadtxt(filename)
                
    def updateweights(self, minbx, minby):
        eta = .03      #learning rate
        d_w1 = np.zeros(self.w1.shape)
        d_w2 = np.zeros(self.w2.shape)
        d_b1 = np.zeros(self.b1.shape)
        d_b2 = np.zeros(self.b2.shape)
        for x, y in zip(minbx, minby):
            dd_w1, dd_b1, dd_w2, dd_b2 = self.backprop(x, y)
            d_w1 = d_w1 + dd_w1
            d_w2 = d_w2 + dd_w2
            d_b1 = d_b1 + dd_b1
            d_b2 = d_b2 + dd_b2
        self.w1 = self.w1 - eta*d_w1
        self.b1 = self.b1 - eta*d_b1
        self.w2 = self.w2 - eta*d_w2
        self.b2 = self.b2 - eta*d_b2

    def backprop(self, x, y):
        d_w1 = np.zeros(self.w1.shape)
        d_w2 = np.zeros(self.w2.shape)
        d_b1 = np.zeros(self.b1.shape)
        d_b2 = np.zeros(self.b2.shape)
        a1 = x
        z2 = np.dot(self.w1, a1) + self.b1
        a2 = self.sigmoid(z2)
        z3 = np.dot(self.w2, a2) + self.b2
        a3 = self.sigmoid(z3)
        delta3 = self.loss_derivative(a3, y)*self.sigprime(z3)
        d_b2 = delta3
        d_w2 = np.dot(delta3, a2.T)
        delta2 = np.dot(self.w2.T, delta3)*self.sigprime(z2)
        d_b1 = delta2
        d_w1 = np.dot(delta2, a1.T)
        return (d_w1, d_b1, d_w2, d_b2)

    def loss_derivative(self, output,target):
        return (output - target)

    def test(self, x):
        ycap = self.feedforward(x)
        #print("ycap: ",ycap)
        return ycap.max()+1

    def sigmoid(self, z):
        fz = 1.0/(1.0+np.exp(-z))
        return fz
    def sigprime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def evaluate(self, testx, testy):
        correct = 0
        total = 0
        for x, y in zip(testx, testy):
            ycap = np.argmax(self.feedforward(x))+1
            total = total+1
            if(ycap == y):
                correct = correct+1
        print("correct: ", correct, "total: ", total, "percentage: ", correct/total)


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    print("train function called")
    net1 = mlp(784, 5, 10)
    net1.sgd(trainX, trainY)
    net1.evaluate(testX, testY)
    #net1.feedforward(trainX)
    #result = net1.test(trainX)
    #print("Digit predited: ",result)


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    #load arrays using w1 = np.loadtxt(filename)
    return np.zeros(testX.shape[0])
