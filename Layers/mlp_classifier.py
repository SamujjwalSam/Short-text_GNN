# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : MLP model.
__description__ :
__project__     : GCPD
__classes__     :
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "10/01/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import torch
# import torch.nn.functional as F
from abc import ABC

from Layers.bilstm_classifiers import BiLSTM_Classifier


class MLP_Model(torch.nn.Module):
    """ Contrastive pretraining model with MLP. """

    def __init__(self, in_dim, hid_dim, out_dim, num_layer=2, dropout=0.2,
                 training=True):
        """MLP model for Pretraining.

        :param in_dim: Input dimension
        :param hid_dim:
        :param out_dim: Output Dimension; should be == n_classes for last layer
        :param num_layer: number of layers (default = 2)
        :param dropout: Feature dropout rate
        :param training:
        """
        super(MLP_Model, self).__init__()
        # self.training = training
        self.dropout = dropout
        # self.in_dim = in_dim
        # self.out_dim = out_dim
        # self.num_layer = num_layer

        self.mlp1 = torch.nn.Linear(in_features=in_dim, out_features=hid_dim)

        self.mlp_layers = []
        for _ in range(num_layer - 2):
            self.mlp_layers.append(torch.nn.Linear(in_features=hid_dim, out_features=hid_dim))

        self.mlp_layers = torch.nn.ModuleList(self.mlp_layers)

        # Final dense layer
        self.mlpn = torch.nn.Linear(in_features=hid_dim, out_features=out_dim)

    def forward(self, X, training=True):
        X = self.mlp1(X)
        # X = F.dropout(X, self.dropout, training=training)
        X = torch.relu(X)

        for mlp_layer in self.mlp_layers:
            X = mlp_layer(X)
            X = torch.relu(X)

        # hidden = [batch size, hid dim]
        X = self.mlpn(X)
        return X


class MLP_BiLSTM_Classifier(torch.nn.Module, ABC):
    """ MLP + BiLSTM classifier """

    def __init__(self, in_dim, hid_dim, out_dim, num_classes, state=None):
        super(MLP_BiLSTM_Classifier, self).__init__()
        self.mlp = MLP_Model(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, dropout=0.2)

        ## Load model state and parameters for the GCN part only:
        if state is not None:
            self.mlp.load_state_dict(state['state_dict'])

        self.bilstm_classifier = BiLSTM_Classifier(out_dim, num_classes)

    def forward(self, X: torch.Tensor, token_global_ids: torch.Tensor,
                save_gcn_embs=False) -> torch.Tensor:
        """ Combines embeddings of tokens using a BiLSTM + Linear Classifier

        :param token_global_ids:
        :param save_gcn_embs:
        :param X: Embeddings from token GCN
        :param A: token graph
        """
        ## Fetch embeddings from token graph
        X = self.mlp(X=X)

        if save_gcn_embs:
            torch.save(X, 'X_gcn.pt')

        ## Fetch embeddings from token graph:
        preds = []
        for node_ids in token_global_ids:
            pred = self.bilstm_classifier(X[node_ids].unsqueeze(0))
            preds.append(pred)

        return torch.stack(preds).squeeze()
