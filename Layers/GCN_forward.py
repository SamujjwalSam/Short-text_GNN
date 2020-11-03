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
import numpy as np
import scipy.sparse as sp

from Utils.utils import sp_coo2torch_coo
from Label_Propagation_PyTorch.adj_propagator import Adj_Propagator


def GCN_forward_old(adj: sp.csr.csr_matrix, X: torch.Tensor, forward: int = 2
                    ) -> torch.Tensor:
    """ Forward pass of GCN.

    :param forward: Number of times GCN multiplication should be applied
    :param adj: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(adj, sp.csr_matrix):
        adj = sp_coo2torch_coo(adj)

    I = sp.eye(*adj.shape).tocsr()
    I = sp_coo2torch_coo(I)

    # I = torch.eye(*adj.shape).type(torch.FloatTensor)
    A_hat = adj + I
    # D = A_hat.sum(dim=0)
    D = torch.sparse.sum(A_hat, dim=0)
    D = D ** -0.5

    D = D.to_dense().numpy()
    D = sp.diags(D).tocsr()
    D = sp_coo2torch_coo(D)

    # D_inv = torch.diag(D_inv).type(torch.FloatTensor)
    A_hat = D * A_hat * D

    for i in range(forward):
        if X.dtype == torch.float64:
            X = torch.spmm(A_hat, X.float())
        else:
            X = torch.spmm(A_hat, X)

    return X


def GCN_forward(adj, X, forward=2):
    """ Forward pass of GCN.

    :param forward: Number of times GCN multiplication should be applied
    :param adj: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(adj, sp.csr_matrix):
        adj = sp_coo2torch_coo(adj)

    adj_normalizer = Adj_Propagator()

    A_hat = adj_normalizer.normalize_adj(adj)

    for i in range(forward):
        if X.dtype == torch.float64:
            X = X.float()

        X = adj_normalizer(A_hat, X)

    return X


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
    output = GCN_forward(G_k_data)
    print(output)


if __name__ == "__main__":
    main()
