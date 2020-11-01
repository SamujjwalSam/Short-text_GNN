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


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ dot product between 2 tensors for dense and sparse format.

    :param x:
    :param y:
    :return:
    """
    if x.is_sparse:
        res = torch.spmm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


class GCN_DropEdgeLearn_Layer(torch.nn.Module):
    """ GCN layer with DropEdge and learnable param matrix S. """
    def __init__(self, num_token, emb_dim, out_dim=2, adj_drop=0.0, bias=True):
        """GCN layer with DropEdge and learnable param matrix S.

        :param num_token: Number of tokens (nodes) in the graph.
        :param emb_dim: Dimension of each node embedding
        :param out_dim: Output Dimension; should be n_classes for last layer
        :param adj_drop: Dropout rate for the adjacency matrix
        :param bias:
        """
        super(GCN_DropEdgeLearn_Layer, self).__init__()
        self.num_token = num_token
        self.weight = Parameter(torch.FloatTensor(emb_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.S = Parameter(torch.FloatTensor(self.num_token, self.num_token))
        self.reset_parameters()

        if adj_drop > 0.0:
            self.apply_adj_dropout(adj_drop=adj_drop)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.S.data)

    def forward(self, A, X):
        torch.autograd.set_detect_anomaly(True)
        A_prime = torch.mul(self.S, A)
        support = torch.mm(X, self.weight)
        X = dot(A_prime, support)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def apply_adj_dropout(self, adj_drop=0.2):
        """ Applies dropout to the whole adjacency matrix.

        :param adj_drop: dropout rate
        """
        ## Apply dropout to whole adjacency matrix:
        # TODO: symmetric dropout (for undirected graph)
        adj_dropout = torch.nn.Dropout(p=adj_drop, inplace=False)
        self.S = adj_dropout(self.S)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_token) + ')'\
               + str(self.emb_dim) + ' -> ' + str(self.out_dim)


class GCN_DropEdgeLearn_Model(torch.nn.Module):
    """ GCN model with DropEdge and learnable param matrix S. """
    def __init__(self, num_token, in_dim, hidden_dim, out_dim, num_layer=2, dropout=0.2,
                 adj_dropout=0.0, training=True):
        """GCN model with DropEdge and learnable param matrix S

        :param num_token: number of tokens
        :param in_dim: Input dimension
        :param hidden_dim:
        :param out_dim: Output Dimension; should be == n_classes for last layer
        :param num_layer: number of layers (default = 2)
        :param dropout: Feature dropout rate
        :param adj_dropout: Dropout rate on the adjacency matrix (default = 0.0)
        :param training:
        """
        super(GCN_DropEdgeLearn_Model, self).__init__()
        self.training = training
        self.dropout = dropout

        self.dropedgelearn_gcn1 = GCN_DropEdgeLearn_Layer(
            num_token, emb_dim=in_dim, out_dim=hidden_dim, adj_drop=adj_dropout)

        self.gcn_layers = []
        for _ in range(num_layer - 2):
            self.gcn_layers.append(GCN_DropEdgeLearn_Layer(
                num_token, emb_dim=hidden_dim, out_dim=hidden_dim, adj_drop=adj_dropout))

        self.gcn_layers = torch.nn.ModuleList(self.gcn_layers)

        # Final dense layer
        self.dropedgelearn_gcnn = GCN_DropEdgeLearn_Layer(
            num_token, emb_dim=hidden_dim, out_dim=out_dim, adj_drop=adj_dropout)

    def forward(self, A, X, targeted_drop_start=0.25):
        """ Combines embeddings of tokens from large and small graph by concatenating.

        Take embeddings from large graph for tokens present in the small graph batch.
        Need token to index information to fetch from large graph.
        Arrange small tokens and large tokens in same order.

        small_batch_graphs: Instance graph batch
        small_batch_embs: Embeddings from instance GAT
        token_idx_batch: Ordered set of token ids present in the current batch of instance graph
        large_graph: Large token graph
        large_embs: Embeddings from large GCN
        combine: How to combine two embeddings (Default: concatenate)

        Should be converted to set of unique tokens before fetching from large graph.
        Convert to boolean mask indicating the tokens present in the current batch of instances.
        Boolean mask size: List of number of tokens in the large graph.
        """
        self.dropedgelearn_gcn.apply_targeted_dropout(
            targeted_drop=targeted_drop_start + self.current_epoch / self.num_epoch)
        X = self.dropedgelearn_gcn1(A, X)
        X = F.dropout(X, self.dropout, training=self.training)

        for gcn_layer in self.gcn_layers:
            X = gcn_layer(A, X)
            X = torch.relu(X)

        # hidden = [batch size, hid dim]
        X = self.dropedgelearn_gcnn(A, X)
        return X

    @staticmethod
    def normalize_adj(A: torch.Tensor, eps: float = 1E-9) -> torch.Tensor:
        """ Normalize adjacency matrix for LPA:
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
    w = 5
    n = w
    emb_dim = 2

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, emb_dim)).float()

    mat_test = GCN_DropEdgeLearn_Layer(w, emb_dim=emb_dim, out_dim=emb_dim)

    trainer = pl.Trainer(max_epochs=epochs)

    trainer.fit(mat_test)
