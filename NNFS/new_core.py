import numpy as np

from image_handling import find_index_range, fetch_image_set
   
class Network(object):
    def __init__(self, sizes, error_type, activation_type):
        self.Layers = []
        self.errorfunc = ErrorFunction(error_type)
        self.activationfunc = ActivationFunction(activation_type)
        for k in range(1, len(sizes)):
            self.Layers.append(Layer(sizes[k - 1], sizes[k], self.errorfunc, self.activationfunc))
    def forward(self, A):
        for Layer in self.Layers:
            A = Layer.forward(A)
        return A
    def backward(self, dE_dY, LR):
        for Layer in self.Layers[::-1]:
            dE_dY = Layer.backward(dE_dY, LR)
    def train_step(self, A, Y, LR):
        N = self.forward(A)
        self.backward(N - Y, LR)
        return N
    
class Layer(object):
    def __init__(self, i, j, error, activation):
        self.errorfunc = error
        self.activationfunc = activation
        self.j = j
        self.i = i
        self.W = np.random.randn(i, j) / np.sqrt(i)
        self.B = np.zeros((1, j))
    def forward(self, X):
        #input is of form (1, i)
        #output is of form (i, j)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.X = X
        self.Z = X @ self.W + self.B
        return self.activationfunc.run(self.Z)
    def backward(self, dE_dY, LR):
        #input is of form of layer
        dE_dZ = dE_dY * self.activationfunc.rundev(self.Z)
        dE_dW = self.X.T @ dE_dZ
        dE_dB = np.sum(dE_dZ, axis=0, keepdims=True)
        self.W -= LR * dE_dW
        self.B -= LR * dE_dB
        dE_dX = dE_dZ @ self.W.T
        return dE_dX

class ActivationFunction():
    def __init__(self, type):
        if (type == "sigmoid"):
            self.type = 0
        elif (type == "tanh"):
            self.type = 1
    def run(self, x):
        if self.type == 0:
            return ActivationFunction.sigmoid(x)
        elif self.type == 1:
            return ActivationFunction.tanh(x)
    def rundev(self, x):
        if self.type == 0:
            return ActivationFunction.dev_sigmoid(x)
        elif self.type == 1:
            return ActivationFunction.dev_tanh(x)
        
    def sigmoid(x):
        return 1/(1 + (np.e ** (-x)))
    def dev_sigmoid(x):
        return ActivationFunction.sigmoid(x) * (1 - ActivationFunction.sigmoid(x))
    def tanh(x):
        return np.tanh(x)
    def dev_tanh(x):
        return 1 - np.tanh(x)**2
    
class ErrorFunction():
    def __init__(self, type):
        if type == "MSE":
            self.type = 0
    def run(self, A, Y):
        if self.type == 0:
            return np.mean((A - Y)**2) / 2
    def rundev(self, A, Y):
        if self.type == 0:
            return (A - Y)

network = Network([2, 3, 1], "MSE", "sigmoid")

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0], [1], [1], [0]])

for epoch in range(0, 500000):
    x = X[epoch % 4]
    y = Y[epoch % 4]
    output = network.forward(x)
    error = network.errorfunc.run(output, y)
    network.train_step(x, y, 0.8)
    if epoch % 100 == 0: 
        print(f"Epoch {epoch} Error: {error}")


print("1:")
f1=network.forward(np.array([0, 0]))
print(f1)
print("2:")
f2=network.forward(np.array([1, 0]))
print(f2)
print("3:")
f3=network.forward(np.array([0, 1]))
print(f3)
print("4:")
f4=network.forward(np.array([1, 1]))
print(f4)