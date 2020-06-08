# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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
