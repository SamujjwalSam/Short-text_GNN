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

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric import utils as tg_utils


def netrowkx2geometric(G):
    G_data = tg_utils.from_networkx(G)
    return G_data


def sp_sparse2torch_sparse(M):
    if not isinstance(M, sp.coo_matrix):
        M = M.tocoo()

    M = torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((M.row, M.col))),
        torch.FloatTensor(M.data),
        torch.Size(M.shape))

    return M


def GCN_forward(adj, X, forward=2):
    """ Forward pass of GCN.

    :param forward: Number of times GCN multiplication should be applied
    :param adj: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(adj, sp.csr_matrix):
        adj = sp_sparse2torch_sparse(adj)

    sp_eye = sp.eye(*adj.shape)
    I = sp_sparse2torch_sparse(sp_eye)

    # I = torch.eye(*adj.shape).type(torch.FloatTensor)
    A_hat = adj + I
    # D = A_hat.sum(dim=0)
    D = torch.sparse.sum(A_hat, dim=0)
    D_inv = D ** -0.5

    D_inv_np = D_inv.to_dense().numpy()
    D_inv_sp = sp.diags(D_inv_np)
    D_inv_t = sp_sparse2torch_sparse(D_inv_sp)

    # D_inv = torch.diag(D_inv).type(torch.FloatTensor)
    A_hat = D_inv_t * A_hat * D_inv_t

    for i in range(forward):
        if X.dtype == torch.float64:
            X = torch.spmm(A_hat, X.float())
        else:
            X = torch.spmm(A_hat, X)

    return X


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
        G_k.nodes[i]['x'] = np.random.rand(7,)
    G_k_data = netrowkx2geometric(G_k)

    print(G_k_data.x)
    print(G_k_data.edge_index)

    gcn_net = GCN_Net_multi_linear(num_node_features=7, hid_dim=5,
                                   num_classes=2, num_gcn_layers=2)

    output = gcn_net(G_k_data)
    print(output)


if __name__ == "__main__":
    main()
