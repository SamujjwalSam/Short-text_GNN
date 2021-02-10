# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GCN with EdgeDrop on adjacency matrix with learnable update
__description__ :
__project__     :
__classes__     :
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "30/10/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from Utils.utils import dot


class GraphConvolutionLayer(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('\
               + str(self.in_features) + ' -> '\
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, A, X):
        X = F.relu(self.gc1(X, A))
        X = F.dropout(X, self.dropout, training=self.training)
        X = self.gc2(X, A)
        # return F.log_softmax(x, dim=1)
        return X


if __name__ == "__main__":
    epochs = 5
    d = 2
    w = 3
    n = d+w
    emb_dim = 2

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, emb_dim)).float()

    mat_test = GCN_DropEdgeLearn_Model(num_docs=d, num_tokens=w, in_dim=2, hidden_dim=2, out_dim=1)

    X_hat = mat_test(A, X)
