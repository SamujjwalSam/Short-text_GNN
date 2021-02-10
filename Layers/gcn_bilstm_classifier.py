# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Forward pass of GCN.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     :
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
import numpy as np
import scipy.sparse as sp

from Layers.gcn_layer import GCN
from Layers.bilstm_classifiers import BiLSTM_Classifier
from Utils.utils import sp_coo2torch_coo
from Label_Propagation_PyTorch.adj_propagator import Adj_Propagator
from Logger.logger import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def GCN_forward_old(adj: sp.csr.csr_matrix, X: torch.Tensor, forward: int = 2
                    ) -> torch.Tensor:
    """ Forward pass of GCN.

    :param forward: Number of times GCN multiplication should be applied
    :param adj: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(adj, sp.csr_matrix):
        adj = sp_coo2torch_coo(adj)

    I = sp.eye(*adj.shape).tocsr()
    I = sp_coo2torch_coo(I)

    # I = torch.eye(*adj.shape).type(torch.FloatTensor)
    A_hat = adj + I
    # D = A_hat.sum(dim=0)
    D = torch.sparse.sum(A_hat, dim=0)
    D = D ** -0.5

    D = D.to_dense().numpy()
    D = sp.diags(D).tocsr()
    D = sp_coo2torch_coo(D)

    # D_inv = torch.diag(D_inv).type(torch.FloatTensor)
    A_hat = D * A_hat * D

    for i in range(forward):
        if X.dtype == torch.float64:
            X = torch.spmm(A_hat, X.float())
        else:
            X = torch.spmm(A_hat, X)

    return X


def GCN_forward(adj, X, forward=2):
    """ Forward pass of GCN.

    :param forward: Number of times GCN multiplication should be applied
    :param adj: Adjacency matrix (#tokens x #tokens)
    :param X: Feature representation (#tokens x emb_dim)
    :return: X' (#tokens x emb_dim)
    """
    if isinstance(adj, sp.csr_matrix):
        adj = sp_coo2torch_coo(adj)

    adj_normalizer = Adj_Propagator()

    A_hat = adj_normalizer.normalize_adj(adj)

    for i in range(forward):
        if X.dtype == torch.float64:
            X = X.float()

        X = adj_normalizer(A_hat, X)

    return X


# class BiLSTM_Classifier(torch.nn.Module):
#     """ BiLSTM with a Linear Layer for classification. """
#
#     # define all the layers used in model
#     def __init__(self, embedding_dim, out_dim, hid_dim=100,
#                  n_layers=2, bidirectional=True, dropout=0.2, num_linear=1):
#         super(BiLSTM_Classifier, self).__init__()
#         self.lstm = torch.nn.LSTM(embedding_dim, hid_dim, num_layers=n_layers,
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
#                 self.linear_layers.append(torch.nn.Linear(hid_dim,
#                                                           hid_dim))
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
#             text:
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


class GCNR_Classifier(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_classes, state=None):
        super(GCNR_Classifier, self).__init__()
        self.token_gcn = GCN(nfeat=in_dim, nhid=hid_dim, nclass=out_dim, dropout=0.2)

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


if __name__ == "__main__":
    test = GCNR_Classifier(in_dim=5, hid_dim=3, out_dim=4, num_classes=2)

    test()



def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    import networkx as nx

    G_k = nx.path_graph(5)
    for i in range(5):
        G_k.nodes[i]['x'] = np.random.rand(7, )
    output = GCN_forward(G_k_data)
    logger.info(output)


if __name__ == "__main__":
    main()
