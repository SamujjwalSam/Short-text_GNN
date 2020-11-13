# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Forward pass of GCN.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     :
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric import utils as tg_utils

from Logger.logger import logger


def netrowkx2geometric(G):
    G_data = tg_utils.from_networkx(G)
    return G_data


class GCN_Net_multi_linear(torch.nn.Module):
    """ Multiple (param) GCN layers. """

    def __init__(self, num_node_features, hid_dim, num_classes,
                 num_gcn_layers=2):
        super(GCN_Net_multi_linear, self).__init__()
        self.gcn_start = GCNConv(num_node_features, hid_dim)

        self.gcn_layers = []
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GCNConv(hid_dim, hid_dim))
        self.gcn_layers = ModuleList(self.gcn_layers)

        self.gcn_final = GCNConv(hid_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.dtype == torch.float64:
            x = x.float()
        else:
            x = x

        x = self.gcn_start(x, edge_index)
        x = F.relu(x)

        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        output = self.gcn_final(x, edge_index)

        return output


class GCN_Net(torch.nn.Module):
    def __init__(self, num_node_features, hid_dim, num_classes):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, hid_dim)
        self.conv2 = GCNConv(hid_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x.dtype == torch.float64:
            x = x.float()
        else:
            x = x
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # return F.log_softmax(x, dim=1)
        return x


def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    import networkx as nx

    G_k = nx.path_graph(5)
    for i in range(5):
        G_k.nodes[i]['x'] = np.random.rand(7, )
    G_k_data = netrowkx2geometric(G_k)

    logger.info(G_k_data.x)
    logger.info(G_k_data.edge_index)

    gcn_net = GCN_Net_multi_linear(num_node_features=7, hid_dim=5,
                                   num_classes=2, num_gcn_layers=2)

    output = gcn_net(G_k_data)
    logger.info(output)


if __name__ == "__main__":
    main()
