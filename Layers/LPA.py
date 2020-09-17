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

import torch as tf
# import numpy as np
# from scipy import sparse
# from sklearn.semi_supervised import LabelSpreading, LabelPropagation
# from imblearn.under_sampling import RandomUnderSampler

from Logger.logger import logger


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


def dot(x: tf.Tensor, y: tf.Tensor, sparse: bool) -> tf.Tensor:
    """ dot product between 2 tensors for dense and sparse format.

    :param x:
    :param y:
    :param sparse:
    :return:
    """
    if sparse:
        res = tf.spmm(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class LPALayer(tf.nn.Module):
    def __init__(self, adj, softmax = tf.nn.Softmax(dim=0)):
        super(LPALayer, self).__init__()
        self.softmax = softmax
        self.adj = self.normalize_adj(adj)

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Y_hat = A * Y

        :param inputs:
        :return:
        """
        output = dot(self.adj, inputs, sparse=True)
        return output

    def normalize_adj(self, adj: tf.Tensor, eps: float = 1E-9) -> tf.Tensor:
        """ Normalize adjacency matrix for LPA:
        A = D^(-1/2) * A * D^(-1/2)
        A = softmax(A)

        :param adj:
        :param eps:
        :return:
        """
        N = adj.shape[0]
        D = adj.sum(0)
        D_sqrt_inv = tf.sqrt(1.0 / (D + eps))
        D1 = tf.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = tf.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * adj * D2
        S = self.softmax(S)

        return S


if __name__ == "__main__":
    adj = tf.rand(4, 4)
    lpa = LPALayer(adj)
    inputs = tf.rand(4, 3)
    outputs = lpa(inputs)
    print(outputs)
