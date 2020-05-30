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


def GCN_forward(A_torch, X):
    """ Forward pass of GCN.

    :param A_torch: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(A_torch, np.matrix):
        A_torch = torch.from_numpy(A_torch)

    # I = torch.eye(*A_torch.shape)
    I = torch.eye(*A_torch.shape).type(torch.FloatTensor)
    A_hat = A_torch + I
    D = A_hat.sum(dim=0)  ## TODO: Values does not contain degree
    ## instead cooccurrence
    D_inv = D ** -0.5
    D_inv = torch.diag(D_inv).type(torch.FloatTensor)
    A_hat = D_inv * A_hat * D_inv
    # A_hat = torch.tensor(A_hat)

    # X = X.double()

    output = torch.spmm(A_hat, X.double())
    return output
