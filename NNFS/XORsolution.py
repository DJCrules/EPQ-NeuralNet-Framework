from new_core import *
from image_handling import *
from visuals import *

def LogicProblem():
    network = Network([2, 3, 1], "MSE", "sigmoid")

    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])

    Y = np.array([[0], [1], [1], [0]])

    for epoch in range(0, 2500):
        x = X[epoch % 4]
        y = Y[epoch % 4]
        output = network.forward(x)
        error = network.errorfunc.run(output, y)
        network.train_step(x, y, 0.7)
        if epoch % 100 == 0: 
            print(f"Epoch {epoch} Error: {error}")


    print("1:")
    f1=np.round(network.forward(np.array([0, 0])))
    print(f1)
    print("2:")
    f2=np.round(network.forward(np.array([0, 1])))
    print(f2)
    print("3:")
    f3=np.round(network.forward(np.array([1, 0])))
    print(f3)
    print("4:")
    f4=np.round(network.forward(np.array([1, 1])))
    print(f4)

LogicProblem()

#example = fetch_image_set(0)[0]

#net = Network([len(example), 7, 7, 4])

#a = [np.random.randn(1) for i in range(net.sizes[0])]

#a = feedforward(net, a)

#multi_graph = multilayered_graph(*net.sizes)
#show_plt_fig(multi_graph)