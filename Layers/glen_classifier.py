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
# import torch.nn.functional as F
from dgl import DGLGraph
# from dgl.nn.pytorch.conv import GATConv, GraphConv

# from Layers.gcn_dropedgelearn import GCN_DropEdgeLearn_Model, GCN
# from Layers.wscp_pretrain import WSCP_Pretrain, WSCP_Layer, contrastive_loss
# from Layers.pretrain_gcn import Pretrain_GCN, GCN_Layer, contrastive_loss
from Pretrain.Layers.pretrain_models import Pretrain_GCN
from Layers.gat_classifiers import Instance_GAT_dgl
from Layers.bilstm_classifiers import BiLSTM_Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class BiLSTM_Classifier(torch.nn.Module):
#     """ BiLSTM for classification. """
#
#     # define all the layers used in model
#     def __init__(self, in_dim, out_dim, hid_dim=100, n_layers=2,
#                  bidirectional=True, dropout=0.2, num_linear=1):
#         super(BiLSTM_Classifier, self).__init__()
#         self.lstm = torch.nn.LSTM(in_dim, hid_dim, num_layers=n_layers,
#                                   bidirectional=bidirectional, dropout=dropout,
#                                   batch_first=True)
#
#         ## Intermediate Linear FC layers, default=0
#         self.linear_layers = []
#         for _ in range(num_linear - 1):
#             if bidirectional:
#                 self.linear_layers.append(torch.nn.Linear(hid_dim * 2,
#                                                           hid_dim * 2))
#             else:
#                 self.linear_layers.append(torch.nn.Linear(hid_dim, hid_dim))
#
#         self.linear_layers = torch.nn.ModuleList(self.linear_layers)
#
#         # Final dense layer
#         if bidirectional:
#             self.fc = torch.nn.Linear(hid_dim * 2, out_dim)
#         else:
#             self.fc = torch.nn.Linear(hid_dim, out_dim)
#
#         # activation function
#         ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
#         # self.act = torch.nn.Sigmoid()
#
#     def forward(self, text, text_lengths=None):
#         """ Takes ids of input text, pads them and predict using BiLSTM.
#
#         Args:
#             text: batch size, seq_len, input dim
#             text_lengths:
#
#         Returns:
#
#         """
#         packed_output, (hidden, cell) = self.lstm(text)
#         # hidden = [batch size, num num_lstm_layers * num directions, hid dim]
#         # cell = [batch size, num num_lstm_layers * num directions, hid dim]
#
#         # concat the final forward and backward hidden state
#         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
#
#         for layer in self.linear_layers:
#             hidden = layer(hidden)
#
#         # hidden = [batch size, hid dim * num directions]
#         logits = self.fc(hidden)
#
#         # Final activation function
#         ## NOTE: Sigmoid not required as BCEWithLogitsLoss calculates sigmoid
#         # logits = self.act(logits)
#
#         return logits


class GLEN_Classifier(torch.nn.Module):
    def __init__(self, num_token, in_dim, hid_dim, num_heads, out_dim, num_classes, combine='concat', state=None):
        super(GLEN_Classifier, self).__init__()
        self.combine = combine
        # self.token_gcn_dropedgelearn = GCN_DropEdgeLearn_Model(
        #     num_token=num_token, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        #     dropout=0.2, adj_dropout=0.0)
        # self.token_gcn = GCN(nfeat=in_dim, nhid=hid_dim, nclass=out_dim, dropout=0.2)
        self.token_gcn = Pretrain_GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)

        ## Load model state and parameters for the GCN part only:
        if state is not None:
            self.token_gcn.load_state_dict(state['state_dict'])

        self.instance_gat_dgl = Instance_GAT_dgl(in_dim=in_dim, hid_dim=hid_dim,
                                                 num_heads=num_heads, out_dim=out_dim)

        if combine == 'concat':
            final_dim = 2 * out_dim
        elif combine == 'avg':
            final_dim = out_dim
        else:
            raise NotImplementedError(f'combine supports either concat or avg.'
                                      f' [{combine}] provided.')
        self.bilstm_classifier = BiLSTM_Classifier(final_dim, num_classes)

    # def train(self, mode: bool = True):
    #     self.token_gcn_dropedgelearn.train()
    #     self.instance_gat_dgl.train()

    def forward(self, instance_batch: DGLGraph, instance_batch_embs: torch.Tensor,
                instance_batch_local_token_ids: list, node_counts: list,
                instance_batch_global_token_ids: list, A: torch.Tensor,
                X: torch.Tensor, save_gcn_embs=False) -> torch.Tensor:
        """ Combines embeddings of tokens from token and instance graph by concatenating.

        Take embeddings from token graph for tokens present in the instance graph batch.

        :param save_gcn_embs:
        :param instance_batch_local_token_ids: local node ids for a instance graph to repeat the node embeddings.
        :param node_counts: As batching graphs lose information about the number of nodes in each instance graph,
        node_counts is a list of nodes (unique) in each instance graph.
        :param combine: How to combine two embeddings (Default: concatenate)
        :param instance_batch_global_token_ids: Ordered set of token ids present in the current instance batch
        :param instance_batch_embs: Embeddings from instance GAT
        :param instance_batch: Instance graph batch
        :param X: Embeddings from token GCN
        :param A: token graph
        """
        ## Fetch embeddings from instance graphs:
        instance_batch_embs = self.instance_gat_dgl(instance_batch, instance_batch_embs)

        ## Fetch embeddings from token graph
        # token_embs = self.token_gcn_dropedgelearn(token_graph, token_embs)
        X = self.token_gcn(A=A, X=X)

        if save_gcn_embs:
            torch.save(X, 'X_gcn.pt')
            # torch.save(X, 'X_glove.pt')

        ## Fetch embeddings from token graph of tokens present in instance batch only:
        ## Fetch consecutive instance embs and global idx from token_embs:
        combined_embs_batch = []
        start_idx = 0
        preds = []
        for instance_local_ids, token_global_ids, node_count in zip(
                instance_batch_local_token_ids, instance_batch_global_token_ids, node_counts):
            end_idx = start_idx + node_count
            # logger.info(start_idx, end_idx, node_count)
            # logger.info(token_embs[token_global_ids].shape, instance_embs_batch[start_idx:end_idx][
            # instance_local_ids].shape)

            ## Combine both embeddings:
            if self.combine == 'concat':
                combined_emb = torch.cat([X[token_global_ids],
                                          instance_batch_embs[start_idx:end_idx][instance_local_ids]], dim=1)
            elif self.combine == 'avg':
                combined_emb = torch.mean(torch.stack(
                    [X[token_global_ids], instance_batch_embs[start_idx:end_idx][instance_local_ids]]), dim=0)
            else:
                raise NotImplementedError(f'combine supports either concat or avg.'
                                          f' [{self.combine}] provided.')
            start_idx = end_idx

            combined_embs_batch.append(combined_emb)

            pred = self.bilstm_classifier(combined_emb.unsqueeze(0))
            preds.append(pred)

        return torch.stack(preds).squeeze()


if __name__ == "__main__":
    test = GLEN_Classifier(in_dim=5, hid_dim=3, num_heads=2, out_dim=4, num_classes=2)

    test()
