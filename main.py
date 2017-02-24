from __future__ import print_function
'''
Deep Learning Programming Assignment 1
--------------------------------------
Name: Mohammad Luqman
Roll No.:   16CS60R52


======================================

Problem Statement:
Implement a simple 1 hidden layer MLP WITHOUT using any deep learning library
for predicting MNIST images. You are allowed to use linear algebra
libraries like numpy.

Resources:
1. https://ift6266h16.wordpress.com/2016/01/11/first-assignment-mlp-on-mnist/
2. https://github.com/tfjgeorge/ift6266/blob/master/notebooks/MLP.ipynb
    (In french. But the same repository has other useful ipython notebooks)

You might want to first code in an ipython notebook and later copy-paste
the code here.



======================================

Instructions:
1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/
    (four files).
2. Extract all the files into a folder named `data' just outside
    the folder containing the main.py file. This code reads the
    data files from the folder '../data'.
3. Complete the functions in the train.py file. You might also
    create other functions for your convenience, but do not change anything
    in the main.py file or the function signatures of the train and test
    functions in the train.py file.
4. The train function must train the neural network given the training
    examples and save the in a folder named `weights' in the same
    folder as main.py
5. The test function must read the saved weights and given the test
    examples it must return the predicted labels.
6. Submit your project folder with the weights. Note: Don't include the
    data folder, which is anyway outside your project folder.

Submission Instructions:
1. Fill your name and roll no in the space provided above.
2. Name your folder in format <Roll No>_<First Name>.
    For example 12CS10001_Rohan
3. Submit a zipped format of the file (.zip only).
'''

import numpy as np
import os
import train


def load_mnist():
    data_dir = '../data'
    np.random.seed(100)

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 784)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 784)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_wrapper():
    trX, trY, teX, teY = load_mnist()
    trX = trX*1.0/255
    teX = teX*1.0/255
    tr_d = (trX, trY)
    te_d = (teX, teY)
    #tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    #validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    #validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, test_data)
    #return (training_data, validation_data, test_data)


def print_digit(digit_pixels, label='?'):
    for i in range(28):
        for j in range(28):
            if digit_pixels[i, j] > 128:
                print('#', end='')
            else:
                print('.', end = '')
        print('\n')

    print('Label: ', label)


def main():
    trainX, trainY, testX, testY = load_mnist()
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data, test_data = load_data_wrapper()
    
    '''Train the network'''
    #Uncomment the following line to re-train the neural network
    train.train(training_data, test_data)
    
    '''Test the network. Weights will be read from the weights folder.'''
    train.test(test_data)
    
    

if __name__ == '__main__':
    main()
