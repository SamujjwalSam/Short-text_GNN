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


class GCN_DropEdgeLearn_Layer(torch.nn.Module):
    """ GCN layer with DropEdge and learnable param matrix S. """
    def __init__(self, num_docs, num_tokens, emb_dim, out_dim=2, bias=True):
        """GCN layer with DropEdge and learnable param matrix S.

        :param num_token: Number of tokens (nodes) in the graph.
        :param emb_dim: Dimension of each node embedding
        :param out_dim: Output Dimension; should be n_classes for last layer
        :param adj_drop: Dropout rate for the adjacency matrix
        :param bias:
        """
        super(GCN_DropEdgeLearn_Layer, self).__init__()
        self.n = num_docs + num_tokens
        self.d = num_docs
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(emb_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.S = Parameter(torch.FloatTensor(self.n, self.n))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.S.data)

    def forward(self, A, X, targeted_drop=0.1):
        torch.autograd.set_detect_anomaly(True)
        D = self.get_targeted_dropout(shape_A=A.shape, targeted_drop=targeted_drop)
        D_prime = torch.mul(self.S, D)
        A_prime = torch.mul(D_prime, A)
        support = torch.mm(X, self.weight)
        X = dot(A_prime, support)

        if self.bias is not None:
            return X + self.bias
        else:
            return X

    def apply_targeted_dropout(self, targeted_drop=0.1):
        """ Applies dropout to the doc-doc portion of adjacency matrix. """
        D = self.get_targeted_dropout(self.A.shape, targeted_drop=targeted_drop)

        for i in range(self.d):
            for j in range(self.d):
                self.S[i, j] *= D[i, j]

    def get_targeted_dropout(self, shape_A, targeted_drop=0.1):
        """ Creates targeted dropout matrix of size A.

        # self.d is number of documents in the Adj matrix
        # shape of doc_mat is (document_count, document_count)
        """
        doc_mat = torch.ones(self.d, self.d)
        targeted_dropout = torch.nn.Dropout(p=targeted_drop, inplace=False)
        doc_mat = targeted_dropout(doc_mat)

        # D is dropout matrix applied to Adjacency matrix:
        D = torch.ones(shape_A)

        for i in range(self.d):
            for j in range(self.d):
                D[i, j] = doc_mat[i, j]

        return D

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

    def apply_adj_dropout(self, adj_drop=0.2):
        """ Applies dropout to the whole adjacency matrix. """
        ## Apply dropout to whole adjacency matrix:
        adj_dropout = torch.nn.Dropout(p=adj_drop, inplace=False)
        self.S = adj_dropout(self.S)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.n) + '): '\
               + str(self.emb_dim) + ' -> ' + str(self.out_dim)


class GCN_DropEdgeLearn_Model(torch.nn.Module):
    """ GCN model with DropEdge and learnable param matrix S. """
    def __init__(self, num_docs, num_tokens, in_dim, hidden_dim, out_dim, num_layer=2, dropout=0.2,
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
        self.num_docs = num_docs
        self.num_tokens = num_tokens
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer

        self.dropedgelearn_gcn1 = GCN_DropEdgeLearn_Layer(
            self.num_docs, self.num_tokens, emb_dim=in_dim, out_dim=hidden_dim)

        self.gcn_layers = []
        for _ in range(num_layer - 2):
            self.gcn_layers.append(GCN_DropEdgeLearn_Layer(
                self.num_docs, self.num_tokens, emb_dim=hidden_dim, out_dim=hidden_dim))

        self.gcn_layers = torch.nn.ModuleList(self.gcn_layers)

        # Final dense layer
        self.dropedgelearn_gcnn = GCN_DropEdgeLearn_Layer(
            self.num_docs, self.num_tokens, emb_dim=hidden_dim, out_dim=out_dim)

    def forward(self, A, X, targeted_drop=0.1):
        """ Combines embeddings of tokens from large and small graph by concatenating.

        Take embeddings from large graph for tokens present in the small graph batch.
        Need token to index information to fetch from large graph.
        Arrange small tokens and large tokens in same order.

        instance_graph_batch: Instance graph batch
        instance_embs_batch: Embeddings from instance GAT
        instance_batch_global_token_ids: Ordered set of token ids present in the current batch of instance graph
        token_graph: Large token graph
        token_embs: Embeddings from large GCN
        combine: How to combine two embeddings (Default: concatenate)

        Should be converted to set of unique tokens before fetching from large graph.
        Convert to boolean mask indicating the tokens present in the current batch of instances.
        Boolean mask size: List of number of tokens in the large graph.
        """
        # self.dropedgelearn_gcn.get_adj_dropout(
        #     targeted_drop=targeted_drop + self.current_epoch / self.num_epoch)
        X = self.dropedgelearn_gcn1(A, X, targeted_drop=targeted_drop)
        X = F.dropout(X, self.dropout, training=self.training)

        for gcn_layer in self.gcn_layers:
            X = gcn_layer(A, X, targeted_drop=targeted_drop)
            X = torch.relu(X)

        # hidden = [batch size, hid dim]
        X = self.dropedgelearn_gcnn(A, X, targeted_drop=targeted_drop)
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
    #     return self.__class__.__name__ + ' (' + str(self.n) + ')'\
    #            + str(self.in_dim) + ' -> ' + str(self.out_dim) + ' x ' + str(self.num_layer)


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
