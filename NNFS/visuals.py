import itertools

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from image_handling import find_images, find_index_range


def show_plt_fig(G):
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(30, 10))
    nx.draw(G, pos, with_labels=False)
    plt.axis("tight")
    plt.show()

def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G

def excel_bar_graph():
    df = pd.read_csv(r"C:\Users\dylan\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3\Data_Entry_2017.csv")
    ID = 0
    findings = 0
    no_findings = 0
    for index, row in df.iterrows():
        if row['Patient ID'] != ID:
            if row['Finding Labels'] != "No Finding":
                findings += 1
            else:
                no_findings += 1
            ID = row['Patient ID']
    print ("not ok: " + str(findings))
    print("ok: " + str(no_findings))
    fig, ax = plt.subplots()
    ax.bar(["Ill", "Not ill"], [findings, no_findings])
    plt.show()

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