from __future__ import print_function
import os
import sys
import random
import numpy as np

class Network(object):

    def __init__(self, indim, hdim, outdim):
        np.random.seed(100)
        #self.num_layers = 2
        self.biases = [np.random.randn(hdim, 1),np.random.randn(outdim, 1)]
        self.weights = [np.random.randn(hdim, indim), np.random.randn(outdim, hdim)]

    def feedforward(self, a):
        """Neural network output"""
        a1 = a
        a2 = self.sigmoid(np.dot(self.weights[0], a1)+self.biases[0])
        a3 = self.sigmoid(np.dot(self.weights[1], a2)+self.biases[1])
        return a3

    def SGD(self, training_data, test_data=None):
        itr = 5                   #number of iterations
        eta = 3.0                 #learning rate
        mini_batch_size = 10      #mini batch size
        if test_data: 
          lentest = len(test_data)
        n = len(training_data)
        for j in range(itr):
            #random.shuffle(training_data)
            mini_batches = []
            for k in range(0, n, mini_batch_size):
                mini_batches.append(training_data[k:k+mini_batch_size])
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print("Iteration",j,": ",self.evaluate(test_data),"/",lentest)
            else:
                print("Iteration: ",j)

    def update_mini_batch(self, mini_batch, eta):
        '''Process each mini batch, compute error and update weights at the end'''
        d_b1 = np.zeros(self.biases[0].shape)
        d_b2 = np.zeros(self.biases[1].shape)
        d_w1 = np.zeros(self.weights[0].shape)
        d_w2 = np.zeros(self.weights[1].shape)

        for x, y in mini_batch:
            dd_b, dd_w = self.backprop(x,y)
            d_b1 = d_b1 + dd_b[0]
            d_b2 = d_b2 + dd_b[1]
            d_w1 = d_w1 + dd_w[0]
            d_w2 = d_w2 + dd_w[1]
       
        eta = eta/len(mini_batch)           #Normalise learning rate by size of mini batch
        self.weights = [self.weights[0]-eta*d_w1, self.weights[1]-eta*d_w2]
        self.biases =  [self.biases[0]-eta*d_b1, self.biases[1]-eta*d_b2]

    def backprop(self, x, y):
        '''For each training example, get output and backpropagate the error'''
        dd_b = [np.zeros(self.biases[0].shape), np.zeros(self.biases[1].shape)]
        dd_w = [np.zeros(self.weights[0].shape), np.zeros(self.weights[1].shape)]
        
        # Feedforward
        a1 = x
        z2 = np.dot(self.weights[0], a1) + self.biases[0]
        a2 = self.sigmoid(z2)
        z3 = np.dot(self.weights[1], a2) + self.biases[1]
        a3 = self.sigmoid(z3)
        
        #Backpropagate
        delta3 = self.loss_derivative(a3, y)*self.sigmoid_prime(z3)
        dd_b[1] = delta3
        dd_w[1] = np.dot(delta3, a2.T)

        delta2 = np.dot(self.weights[1].T, delta3)*self.sigmoid_prime(z2)
        dd_b[0] = delta2
        dd_w[0] = np.dot(delta2, a1.T)

        return (dd_b, dd_w)

    def evaluate(self, test_data):
        '''return the number of inputs correctly classified'''
        results = np.zeros((len(test_data),1))
        correct = 0
        for x,y in test_data:
          output = np.argmax(self.feedforward(x))
          if(output==y):
            correct = correct + 1
        return correct

    def loss_derivative(self, output, y):
        '''Return dL/dy. L is mean squared function'''
        return (output-y)

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        '''Derivative of the sigmoid function'''
        return self.sigmoid(z)*(1-self.sigmoid(z))



