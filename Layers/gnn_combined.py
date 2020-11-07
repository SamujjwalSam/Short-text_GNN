# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GNN Wrapper to combine token token graph with instance graphs.
__description__ : GNN Wrapper to combine token token graph with instance graphs written in DGL library
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "20/10/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GATConv, GraphConv

from Layers.gcn_dropedgelearn import GCN_DropEdgeLearn_Model


class Instance_GAT_dgl(torch.nn.Module):
    """ GAT architecture for instance graphs. """

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, out_dim: int):
        super(Instance_GAT_dgl, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g: DGLGraph, emb: torch.Tensor = None) -> torch.Tensor:
        if emb is None:
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        # g.ndata['emb'] = emb
        return emb


# class Token_Graph_GCN(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super(Token_Graph_GCN, self).__init__()
#         self.conv1 = GraphConv(in_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, out_dim)
#
#     def forward(self, g, emb):
#         if emb is None:
#             emb = g.ndata['emb']
#         emb = self.conv1(g, emb)
#         emb = torch.relu(emb)
#         emb = self.conv2(g, emb)
#         return emb


class BiLSTM_Classifier(torch.nn.Module):
    """ BiLSTM for classification. """

    # define all the layers used in model
    def __init__(self, hidden_dim, output_dim, embedding_dim=100,
                 n_layers=2, bidirectional=True, dropout=0.2, num_linear=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                                  bidirectional=bidirectional, dropout=dropout,
                                  batch_first=True)

        ## Intermediate Linear FC layers, default=0
        self.linear_layers = []
        for _ in range(num_linear - 1):
            if bidirectional:
                self.linear_layers.append(torch.nn.Linear(hidden_dim * 2,
                                                          hidden_dim * 2))
            else:
                self.linear_layers.append(torch.nn.Linear(hidden_dim,
                                                          hidden_dim))

        self.linear_layers = torch.nn.ModuleList(self.linear_layers)

        # Final dense layer
        if bidirectional:
            self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # self.act = torch.nn.Sigmoid()

    def forward(self, text, text_lengths):
        """ Takes ids of input text, pads them and predict using BiLSTM.

        Args:
            text:
            text_lengths:

        Returns:

        """
        packed_output, (hidden, cell) = self.lstm(text)
        # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
        # cell = [batch size, num num_lstm_layers * num directions, hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        for layer in self.linear_layers:
            hidden = layer(hidden)

        # hidden = [batch size, hid dim * num directions]
        logits = self.fc(hidden)

        # Final activation function
        ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
        # logits = self.act(logits)

        return logits


class GNN_Combined(torch.nn.Module):
    def __init__(self, num_token, in_dim, hidden_dim, num_heads, out_dim, num_classes, combine='concat'):
        super(GNN_Combined, self).__init__()
        self.combine = combine
        self.token_gcn = GCN_DropEdgeLearn_Model(num_token=num_token, in_dim=in_dim, hidden_dim=hidden_dim,
                                                 out_dim=out_dim, dropout=0.2, adj_dropout=0.0)

        self.instance_gat_dgl = Instance_GAT_dgl(in_dim=in_dim, hidden_dim=hidden_dim,
                                                 num_heads=num_heads, out_dim=out_dim)

        if combine == 'concat':
            final_dim = 2 * out_dim
        elif combine == 'avg':
            final_dim = out_dim
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        self.bilstm_classifier = BiLSTM_Classifier(final_dim, num_classes)

    def forward(self, instance_batch, instance_batch_embs, instance_batch_token_ids,
                token_graph, token_embs):
        """ Combines embeddings of tokens from token and instance graph by concatenating.

        Take embeddings from token graph for tokens present in the instance graph batch.

        :param combine: How to combine two embeddings (Default: concatenate)
        :param instance_batch_token_ids: Ordered set of token ids present in the current instance batch
        :param instance_batch_embs: Embeddings from instance GAT
        :param instance_batch: Instance graph batch
        :param token_embs: Embeddings from token GCN
        :param token_graph: token graph
        """
        ## Fetch embeddings from instance graphs:
        instance_batch_embs = self.instance_gat_dgl(instance_batch, instance_batch_embs)

        ## Fetch embeddings from token graph
        token_embs = self.token_gcn(token_graph, token_embs)

        ## Fetch embeddings from token graph of tokens present in instance batch only:
        token_embs = token_embs[instance_batch_token_ids]

        ## Combine both embeddings:
        if self.combine == 'concat':
            embs = torch.cat([instance_batch_embs, token_embs])
        elif self.combine == 'avg':
            embs = torch.mean(torch.stack([instance_batch_embs, token_embs]), dim=0)
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{self.combine}] provided.')
        return self.bilstm_classifier(embs)


if __name__ == "__main__":
    test = GNN_Combined(in_dim=5, hidden_dim=3, num_heads=2, out_dim=4, num_classes=2)

    test()
