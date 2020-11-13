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

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


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
        self.emb_dim = emb_dim
        self.out_dim = out_dim
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
        D = self.get_adj_dropout(shape_A=A.shape)
        D_prime = torch.mul(self.S, D)
        A_prime = torch.mul(D_prime, A)
        support = torch.mm(X, self.weight)
        X = dot(A_prime, support)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def get_adj_dropout(self, shape_A, adj_drop=0.2):
        """ Get dropout matrix of shape adjacency matrix

        TODO: symmetric dropout (for undirected graph)

        :param adj_drop: dropout rate
        """
        ## Get dropout matrix of shape adjacency matrix:
        adj_dropout = torch.nn.Dropout(p=adj_drop, inplace=True)
        D_drop = torch.ones(shape_A)
        adj_dropout(D_drop)
        return D_drop

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_token) + '): '\
               + str(self.emb_dim) + ' -> ' + str(self.out_dim)


class GCN_DropEdgeLearn_Model(torch.nn.Module):
    """ GCN model with DropEdge and learnable param matrix S. """
    def __init__(self, num_token, in_dim, hidden_dim, out_dim, num_layer=2, dropout=0.2,
                 adj_dropout=0.1, training=True):
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
        self.num_token = num_token
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer

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

        instance_batch: Instance graph batch
        instance_batch_embs: Embeddings from instance GAT
        instance_batch_global_token_ids: Ordered set of token ids present in the current batch of instance graph
        token_graph: Large token graph
        token_embs: Embeddings from large GCN
        combine: How to combine two embeddings (Default: concatenate)

        Should be converted to set of unique tokens before fetching from large graph.
        Convert to boolean mask indicating the tokens present in the current batch of instances.
        Boolean mask size: List of number of tokens in the large graph.
        """
        # self.dropedgelearn_gcn.get_adj_dropout(
        #     targeted_drop=targeted_drop_start + self.current_epoch / self.num_epoch)
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

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.num_token) + ')'\
    #            + str(self.in_dim) + ' -> ' + str(self.out_dim) + ' x ' + str(self.num_layer)


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
