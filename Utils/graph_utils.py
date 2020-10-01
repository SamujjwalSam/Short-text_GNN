# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7

"""
__synopsis__    : Saves and loads DGL graphs
__description__ : Uses DGL save and loader
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import networkx as nx
import matplotlib.pyplot as plt
from dgl.data.utils import save_graphs, load_graphs


def save_dgl_graphs(path, graphs, labels_dict=None):
    """ Saves DGL graphs along with associated labels list.

    :param path:
    :param graphs:
    :param labels_dict: Should be a dict, does not support Tensor directly.
    """
    save_graphs(str(path), graphs, labels_dict)


def load_dgl_graphs(path, idx_list=None):
    """ Loads DGL graphs and their labels.

    :param path:
    :param idx_list:
    :return:
    """
    graphs, labels = load_graphs(path, idx_list)
    labels_tensor_list = labels["glabel"]
    labels_list = []
    for label in labels_tensor_list:
        labels_list += [label.tolist()]
    return graphs, labels_list


def visualize_dgl_graph_as_networkx(graph):
    """ Plot DGL graph as networkx.

    :param graph:
    """
    graph = graph.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(graph)
    # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_label=True)
    plt.show()
