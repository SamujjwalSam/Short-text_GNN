# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Label Propagation using PyTorch
__description__ : Implementing Label Propagation in PyTorch for continuous label
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

import torch as t

# import numpy as np
# from scipy import sparse
# from sklearn.semi_supervised import LabelSpreading, LabelPropagation
# from imblearn.under_sampling import RandomUnderSampler

# class SparseDropout(tf.nn.Module):
#     def __init__(self, dprob=0.5):
#         super(SparseDropout, self).__init__()
#         # dprob is ratio of dropout
#         # convert to keep probability
#         self.kprob = 1 - dprob
#
#     def forward(self, x):
#         mask = ((tf.rand(x._values().size()) + self.kprob).floor()).type(
#             tf.bool)
#         rc = x._indices()[:, mask]
#         val = x._values()[mask] * (1.0 / self.kprob)
#         return tf.sparse.FloatTensor(rc, val)


# def sparse_dropout(x, keep_prob, noise_shape):
#     random_tensor = keep_prob
#     # random_tensor += tf.random_uniform([noise_shape], dtype=tf.float64)
#     random_tensor += tf.rand([noise_shape]).float()
#     # dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
#     dropout_mask = tf.floor(random_tensor).bool()
#     # res = tf.sparse_retain(x, dropout_mask)
#     sp_dropout = SparseDropout(keep_prob)
#     res = sp_dropout(x, dropout_mask)
#     res /= keep_prob
#     return res


def dot(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """ dot product between 2 tensors for dense and sparse format.

    :param x:
    :param y:
    :return:
    """
    if x.is_sparse:
        res = t.spmm(x, y)
    else:
        res = t.matmul(x, y)
    return res


class Adj_Propagator(t.nn.Module):
    def __init__(self) -> None:
        super(Adj_Propagator, self).__init__()
        # self.softmax = t.nn.Softmax(dim=0)
        # self.labelled_mask = labelled_mask

    def forward(self, adj: t.Tensor, Y: t.Tensor) -> t.Tensor:
        """ Y_hat = A * Y

        :param adj:
        :param Y:
        :param inputs:
        :return:
        """
        adj = self.normalize_adj(adj)
        Y_hat = dot(adj, Y)
        # Y_hat[self.labelled_mask] = Y[self.labelled_mask]
        return Y_hat

    @staticmethod
    def normalize_adj(adj: t.Tensor, eps: float = 1E-9) -> t.Tensor:
        """ Normalize adjacency matrix for LPA:
        A = D^(-1/2) * A * D^(-1/2)
        A = softmax(A)

        :param adj: adjacency matrix
        :param eps: small value
        :return:
        """
        D = t.sparse.sum(adj, dim=0)
        D = t.sqrt(1.0 / (D.to_dense() + eps))
        D = t.diag(D).to_sparse()
        # nz_indices = t.nonzero(D, as_tuple=False)
        # D = t.sparse.FloatTensor(nz_indices.T, D, adj.shape)
        adj = dot(D, adj.to_dense()).to_sparse()
        adj = dot(adj, D.to_dense()).to_sparse()
        # adj = self.softmax(adj)

        return adj


if __name__ == "__main__":
    adj = t.rand(4, 4)
    lpa = Adj_Propagator()
    Y = t.rand(4, 3)
    Y_hat = lpa(adj, Y)
    print(Y_hat)
