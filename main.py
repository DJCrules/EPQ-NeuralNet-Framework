from NNFS.core import *
from NNFS.image_handling import *
from NNFS.visuals import *

net = Network([15, 7, 7, 4])

a = [np.random.randn(1) for i in range(net.sizes[0])]

image_list = (fetch_image_set(0))

print(cost([1 for j in range(net.sizes[2])], feedforward(net, a)))

multi_graph = multilayered_graph(*net.sizes)
show_plt_fig(multi_graph)