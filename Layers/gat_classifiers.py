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
from abc import ABC

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GATConv, GraphConv

from Layers.bilstm_classifiers import BiLSTM_Classifier


class Instance_GAT_dgl(torch.nn.Module, ABC):
    """ GAT architecture for instance graphs. """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_heads: int):
        super(Instance_GAT_dgl, self).__init__()

        self.conv1 = GATConv(in_dim, hid_dim, num_heads,
                             # feat_drop=0.1, attn_drop=0.1, residual=True
                             )
        self.conv2 = GATConv(hid_dim * num_heads, out_dim, 1)

    def forward(self, g: DGLGraph, emb: torch.Tensor = None) -> torch.Tensor:
        if emb is None:
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        # emb = F.relu(self.conv2(g, emb))
        emb = self.conv2(g, emb)
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        g.ndata['emb'] = emb
        return emb


class GAT_BiLSTM_Classifier(torch.nn.Module, ABC):
    """ GAT model with optional BiLSTM """
    def __init__(self, in_dim: int, hid_dim: int, num_heads: int, out_dim: int,
                 lstm_layer=True) -> None:
        super(GAT_BiLSTM_Classifier, self).__init__()
        self.lstm_layer = lstm_layer

        if self.lstm_layer:
            self.bilstm_classifier = BiLSTM_Classifier(hid_dim, out_dim)
            ## Change GAT out_dim size based on LSTM:
            gat_out = hid_dim
        else:
            gat_out = out_dim

        self.instance_gat_dgl = Instance_GAT_dgl(in_dim=in_dim, hid_dim=hid_dim,
                                                 out_dim=gat_out, num_heads=num_heads)

    def forward(self, instance_graph_batch: DGLGraph, instance_embs_batch: torch.Tensor,
                instance_batch_local_token_ids: list, node_counts: list) -> torch.Tensor:
        """ Instance graph classification using GAT.

        :param instance_batch_local_token_ids:
        :param node_counts:
        :param instance_embs_batch: Embeddings from instance GAT
        :param instance_graph_batch: Instance graph batch
        """
        ## Fetch embeddings from instance graphs:
        instance_embs_batch = self.instance_gat_dgl(instance_graph_batch, instance_embs_batch)

        preds = []
        start_idx = 0
        for instance_local_ids, node_count in zip(instance_batch_local_token_ids, node_counts):
            end_idx = start_idx + node_count
            pred = instance_embs_batch[start_idx:end_idx][instance_local_ids]

            if self.lstm_layer:
                pred = self.bilstm_classifier(pred.unsqueeze(0))
            preds.append(pred)

            start_idx = end_idx

        return torch.stack(preds).squeeze()


if __name__ == "__main__":
    test = GAT_BiLSTM_Classifier(in_dim=5, hid_dim=3, out_dim=4, num_heads=2)

    test()
