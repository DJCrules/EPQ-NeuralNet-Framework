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

def cost(desired_output, network_output):
    totalSum = 0
    for output in network_output:
        totalSum += abs(desired_output - network_output) ** 2
    return totalSum / (2 * len(desired_output))

def sigmoid(x):
    return 1/(1 + (math.e ** (-x)))

def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def back_propagate(self):

    return self

def gradient_descent(self, fileName):
    """
    :param self:
    :param string fileName:
    :return:
    """

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


