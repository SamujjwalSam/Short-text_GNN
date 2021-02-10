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
from abc import ABC

from Utils.utils import dot
from Layers.bilstm_classifiers import BiLSTM_Classifier


class GCN_BiLSTM_Classifier(torch.nn.Module, ABC):
    """ GCN + BiLSTM classifier """
    def __init__(self, in_dim, hid_dim, out_dim, num_classes, state=None):
        super(GCN_BiLSTM_Classifier, self).__init__()
        self.token_gcn = GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, dropout=0.2)

        ## Load model state and parameters for the GCN part only:
        if state is not None:
            self.token_gcn.load_state_dict(state['state_dict'])

        self.bilstm_classifier = BiLSTM_Classifier(out_dim, num_classes)

    def forward(self, A: torch.Tensor, X: torch.Tensor,
                token_global_ids: torch.Tensor, save_gcn_embs=False) -> torch.Tensor:
        """ Combines embeddings of tokens using a BiLSTM + Linear Classifier

        :param token_global_ids:
        :param save_gcn_embs:
        :param X: Embeddings from token GCN
        :param A: token graph
        """
        ## Fetch embeddings from token graph
        X = self.token_gcn(A=A, X=X)

        if save_gcn_embs:
            torch.save(X, 'X_gcn.pt')

        ## Fetch embeddings from token graph:
        preds = []
        for node_ids in token_global_ids:
            pred = self.bilstm_classifier(X[node_ids].unsqueeze(0))
            preds.append(pred)

        return torch.stack(preds).squeeze()


class GCN_Layer(torch.nn.Module):
    """ GCN layer for contrastive pretraining. """

    def __init__(self, in_dim, out_dim=2, bias=True):
        """GCN layer with DropEdge and learnable param matrix S.

        :param in_dim: Dimension of each node embedding
        :param out_dim: Output Dimension; should be n_classes for last layer
        :param bias:
        """
        super(GCN_Layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(self.in_dim, self.out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, A, X):
        torch.autograd.set_detect_anomaly(True)
        X = torch.mm(X, self.weight)
        X = dot(A, X)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> '\
               + str(self.out_dim) + ')'


class GCN(torch.nn.Module):
    """ GCN contrastive pretraining model. """

    def __init__(self, in_dim, hid_dim, out_dim, num_layer=2, dropout=0.2, training=True):
        """GCN model with DropEdge and learnable param matrix S

        :param in_dim: Input dimension
        :param hid_dim:
        :param out_dim: Output Dimension; should be == n_classes for last layer
        :param num_layer: number of layers (default = 2)
        :param dropout: Feature dropout rate
        :param training:
        """
        super(GCN, self).__init__()
        self.training = training
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer

        self.gcn_1 = GCN_Layer(in_dim=in_dim, out_dim=hid_dim)

        self.gcn_i = []
        for _ in range(num_layer - 2):
            self.gcn_i.append(GCN_Layer(in_dim=hid_dim, out_dim=hid_dim))

        self.gcn_i = torch.nn.ModuleList(self.gcn_i)

        # Final dense layer
        self.gcn_n = GCN_Layer(in_dim=hid_dim, out_dim=out_dim)

    def forward(self, A, X, training=True):
        """ Combines embeddings of tokens from large and small graph by concatenating.

        Take embeddings from large graph for tokens present in the small graph batch.
        Need token to index information to fetch from large graph.
        Arrange small tokens and large tokens in same order.

        token_graph: Large token graph
        token_embs: Embeddings from large GCN

        Should be converted to set of unique tokens before fetching from large graph.
        Convert to boolean mask indicating the tokens present in the current batch of instances.
        Boolean mask size: List of number of tokens in the large graph.
        """
        X = self.gcn_1(A, X)
        X = F.dropout(X, self.dropout, training=training)

        for gcn in self.gcn_i:
            X = gcn(A, X)
            X = torch.relu(X)

        # hidden = [batch size, hid dim]
        X = self.gcn_n(A, X)
        return X

    @staticmethod
    def normalize_adj(A: torch.Tensor, eps: float = 1E-9) -> torch.Tensor:
        """ Normalize adjacency matrix.

        A = D^(-1/2) * A * D^(-1/2)
        A = softmax(A)

        :param A: adjacency matrix
        :param eps: small value
        :return:
        """
        D = torch.sparse.sum(A, dim=0)
        D = torch.sqrt(1.0 / (D.to_dense() + eps))
        D = torch.diag(D).to_sparse()
        # nz_indices = torch.nonzero(D, as_tuple=False)
        # D = torch.sparse.FloatTensor(nz_indices.T, D, adj.shape)
        A = dot(D, A.to_dense()).to_sparse()
        A = dot(A, D.to_dense()).to_sparse()
        # A = self.softmax(A)

        return A


if __name__ == "__main__":
    epochs = 5
    d = 2
    w = 3
    n = d+w
    emb_dim = 2

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, emb_dim)).float()

    mat_test = GCN(in_dim=2, hid_dim=2, out_dim=1)

    X_hat = mat_test(A, X)
