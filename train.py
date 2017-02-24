'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Mohammad Luqman
Roll No.: 16CS60R52

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import os
import sys
import numpy as np
import network

def train_backup(trainX, trainY):
    '''
    Complete this function.
    print("train function called")
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(784, 15, 10)
    net.SGD(training_data, test_data=None)
    #net1.feedforward(trainX)
    #result = net1.test(trainX)
    #print("Digit predited: ",result)
    '''

def train(training_data, test_data):
    '''Build and train a network with 35 units in the hidden layer'''
    training_data, test_data = load_data_wrapper()
    net = network.Network(784, 35, 10)
    net.SGD(training_data, test_data)
    i = 1
    for w in net.weights:
        np.save("weights/w"+str(i)+".npy", w)
        i+=1
    i = 1
    for b in net.biases:
        np.save("weights/b"+str(i)+".npy", b)
        i+=1
    
def test(test_data):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    net = network.Network(784, 35, 10)          #35 units in the hidden layer

    '''Load weights from the weights folder'''

    net.weights[0] = np.load("weights/w1.npy")
    net.weights[1] = np.load("weights/w2.npy")
    net.biases[0]  = np.load("weights/b1.npy")
    net.biases[1]  = np.load("weights/b2.npy")
    #load arrays using w1 = np.loadtxt(filename)
    labels = net.test(test_data)
    #return labels
