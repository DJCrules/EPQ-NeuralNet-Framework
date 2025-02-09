import math
from random import randint
import pandas as pd
import numpy as np

from image_handling import find_index_range, fetch_image_set


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

def gradient_descend(self, batchsize, imageset):
    # find and store the important bits
    # feed the data forward
    # compare the output to the expected output
    # backpropogate the error through the network
    # repeat.
    answers = convert_meta_data(get_meta_data(imageset))
    batchnumber = (len(answers) / batchsize) + 1
    images = fetch_image_set(0)
    #for i in range (0, batchnumber):
        #for j in range(0, batchsize):
    return self

def single_training_example(self, image_array):
    exit_activations = feedforward(self, image_array)
    


def get_meta_data(set_num):
    index_range = find_index_range(set_num)
    a = 0
    data = []
    df = pd.read_csv(r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3\Data_Entry_2017.csv")
    for index, row in df.iterrows():
        if str(row['Image Index']) == index_range[0]:
            a = 1
        if str(row['Image Index']) == index_range[1]:
            return data
        if a == 1:
            data.append(row["Finding Labels"])
    return data

def convert_meta_data(data):
    #Organisation of metadata:
    #[Cardiomegaly, Emphysema, Effusion, Hernia, Infiltration, Nodule,
    #  Mass, Pneumothorax, Pleural_Thickening, Atelectasis, Fibrosis, 
    #  Consolidation, Edema]
    converted_data = []
    i = 0
    for part in data:
        converted_data.append([])
        if 'Cardiomegaly' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Emphysema' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Effusion' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Hernia' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Infiltration' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Nodule' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Mass' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Pneumothorax' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Pleural_Thickening' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Atelectasis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Fibrosis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Consolidation' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Edema' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'Atelectasis' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        if 'No Finding' in part:
            converted_data[i].append(1)
        else:
            converted_data[i].append(0)
        i+=1
    return converted_data
        
        