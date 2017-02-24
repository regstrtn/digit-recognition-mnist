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
import network


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    print("train function called")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(784, 15, 10)
    net.SGD(training_data, test_data=None)
    #net1.feedforward(trainX)
    #result = net1.test(trainX)
    #print("Digit predited: ",result)

def trainbook(training_data, test_data):
    net = network.Network(784, 15, 10)
    net.SGD(training_data, test_data=test_data)
    i = 1
    for w in net.weights:
        np.savetxt("weights/w"+str(i)+".txt", w)
        i+=1
    i = 1
    for b in net.biases:
        np.savetxt("weights/b"+str(i)+".txt", b)
        i+=1
    

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
    net = network.Network(784, 15, 10)
    net.weights[0] = np.loadtxt("w1.txt")
    net.weights[1] = np.loadtxt("w2.txt")
    net.biases[0]  = np.loadtxt("b1.txt")
    net.biases[1]  = np.loadtxt("b2.txt")
    #load arrays using w1 = np.loadtxt(filename)
    labels = np.zeroes((len(test_data),1))
    return np.zeros(testX.shape[0])
