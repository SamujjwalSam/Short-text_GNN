# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : GNN Wrapper to combine large token graph with small instance graphs.
__description__ : GNN Wrapper to combine large token graph with small instance graphs written in DGL library
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


class Instance_Graphs_GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, out_dim: int) -> None:
        super(Instance_Graphs_GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g: DGLGraph, emb: torch.Tensor = None) -> (DGLGraph, torch.Tensor):
        if emb is None:
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        g.ndata['emb'] = emb
        return g, emb


class Token_Graph_GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Token_Graph_GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, emb):
        if emb is None:
            # Use node degree as the initial node feature. For undirected graphs,
            # the in-degree is the same as the out_degree.
            # emb = g.in_degrees().view(-1, 1).float()
            emb = g.ndata['emb']
        emb = self.conv1(g, emb)
        emb = torch.relu(emb)
        emb = self.conv2(g, emb)
        return emb


class GNN_Combined(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim, num_classes):
        super(GNN_Combined, self).__init__()
        self.large_token_gcn = Token_Graph_GCN(in_dim=in_dim, hidden_dim=hidden_dim,
                                               out_dim=out_dim)
        self.small_instance_gat = Instance_Graphs_GAT(in_dim=in_dim, hidden_dim=hidden_dim,
                                                      num_heads=num_heads, out_dim=out_dim)
        ## We may have different out_dim's from 2 GNNs and concat them.

        self.classify = torch.nn.Linear(2 * out_dim, num_classes)

    def forward(self, small_batch_graphs, small_batch_embs, token_idx_batch,
                large_graph, large_embs, combine='concat'):
        """ Combines embeddings of tokens from large and small graph by concatenating.

        Take embeddings from large graph for tokens present in the small graph batch.
        Need token to index information to fetch from large graph.
        Arrange small tokens and large tokens in same order.

        :param combine: How to combine two embeddings (Default: concatenate)
        :param token_idx_batch: token indices present in current instance graph batch.
        Should be boolean mask indicating the tokens present in the current batch of instances.
        List of size: number of tokens in the large graph.
        :param small_batch_embs: Embeddings from instance GAT
        :param small_batch_graphs: Instance graph batch
        :param large_embs: Embeddings from large GCN
        :param large_graph: Large token graph
        """
        ## Fetch embeddings from instance graphs:
        token_embs_small = self.small_instance_gat(small_batch_graphs, small_batch_embs)

        ## Fetch embeddings from large token graph
        token_embs_large = self.large_token_gcn(large_graph, large_embs)

        ## Fetch embeddings from large graph of tokens present in instance batch only:
        token_embs_large = token_embs_large[token_idx_batch]

        ## Replicating embeddings for as per the order of tokens in instance batch:
        # TODO: Repeat embeddings to get same dim.

        ## Combines both embeddings:
        if combine == 'concat':
            embs = torch.cat([token_embs_small, token_embs_large])
        elif combine == 'avg':
            embs = torch.mean(torch.stack([token_embs_small, token_embs_large]), dim=0)
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        return self.classify(embs)


if __name__ == "__main__":
    test = GNN_Combined(in_dim=5, hidden_dim=3, num_heads=2, out_dim=4, num_classes=2)

    test()
