from core import *
from image_handling import *
from visuals import *

example = fetch_image_set(0)[0]

net = Network([len(example), 7, 7, 4])
single_training_example(net, example)

#a = [np.random.randn(1) for i in range(net.sizes[0])]

#a = feedforward(net, a)




#multi_graph = multilayered_graph(*net.sizes)
#show_plt_fig(multi_graph)