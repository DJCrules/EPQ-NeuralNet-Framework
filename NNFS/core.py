import math
from random import randint

import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

def quadratic_cost(desired_output, network_output):
    totalSum = 0
    for d_output, n_output in desired_output, network_output:
        totalSum += abs(d_output - n_output) ** 2
    return totalSum / (2 * len(desired_output))

def derivative_quadratic_cost(desired_output, network_output):
    return (network_output - desired_output)

def sigmoid(x):
    return 1/(1 + (math.e ** (-x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feedforward(self, activations):
    for bias, weight in zip(self.biases, self.weights):
        activations = sigmoid(np.dot(weight, activations) + bias)
    return activations

def back_propagate(self, activations, learning_rate, answers):
    # take the derivitive of the cost function with respect to the acitvations in the first layer
    # use the chain rule
    # recurse through the network
    
    return self

def gradient_descent(self, fileName, batchsize):
    # choose dataset
    # find and store the important bits
    # feed the data forward
    # compare the output to the expected output
    # backpropogate the error through the network
    # repeat.

    return self

def find_metadata(set_num, specific = "*"):
    return set_num

def randomise(arr):
    for i in range(len(arr) - 1, 0, -1):
        j = randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
        if j > len(arr):
            print("randomised")
    return arr


