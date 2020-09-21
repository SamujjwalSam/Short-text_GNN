# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Code related to applying GNN from DGL library
__description__ : DGL node and graph classification
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "05/08/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dgl import DGLGraph, batch as g_batch, mean_nodes
from dgl.nn.pytorch.conv import GATConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Logger.logger import logger


def plot_graph(g):
    plt.subplot(122)
    nx.draw(g.to_networkx(), with_labels=True)

    plt.show()


class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GATModel, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        ## Concatenating multiple head embeddings
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)
        h = self.layer2(g, h).squeeze()
        return h


def train_node_classifier(g, features, labels, labelled_mask, model, loss_func,
                          optimizer, epochs=5):
    model.train()
    dur = []
    for epoch in range(epochs):
        t0 = time.time()
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = loss_func(logp[labelled_mask], labels[labelled_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        logger.info("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))


def node_binary_classification(hid_feats=4, out_feats=7, num_heads=2):
    from dgl.data import citation_graph as citegrh

    def load_cora_data():
        data = citegrh.load_cora()
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.BoolTensor(data.train_mask)
        g = DGLGraph(data.graph)
        return g, features, labels, mask

    g, features, labels, mask = load_cora_data()

    net = GATModel(in_dim=features.size(1), hidden_dim=hid_feats,
                   out_dim=out_feats, num_heads=num_heads)
    logger.info(net)

    loss_func = F.nll_loss

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train_node_classifier(g, features, labels, labelled_mask=mask, model=net,
                          loss_func=loss_func, optimizer=optimizer, epochs=5)


def test_node_classifier():
    pass


def batch_graphs(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = g_batch(graphs)
    return batched_graph, torch.tensor(labels)


class GAT_Graph_Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GAT_Graph_Classifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.classify = torch.nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, g, h=None):
        if h is None:
            # Use node degree as the initial node feature. For undirected graphs,
            # the in-degree is the same as the out_degree.
            h = g.in_degrees().view(-1, 1).float()

        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = h.view(-1, h.size(1) * h.size(2)).float()
        h = F.relu(self.conv2(g, h))
        h = h.view(-1, h.size(1) * h.size(2)).float()
        g.ndata['h'] = h

        # Calculate graph representation by averaging all node representations.
        hg = mean_nodes(g, 'h')
        return self.classify(hg)


def train_graph_classifier(model, data_loader, loss_func, optimizer, epochs=5):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iter, (graph_batch, label) in enumerate(data_loader):
            prediction = model(graph_batch)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        logger.info('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)


def test_graph_classifier():
    pass


def graph_multiclass_classification(in_feats=1, hid_feats=4, num_heads=2):
    from dgl.data import MiniGCDataset

    # Create training and test sets.
    trainset = MiniGCDataset(320, 10, 20)
    testset = MiniGCDataset(80, 10, 20)

    # # Use PyTorch's DataLoader and the collate function defined before.
    data_loader = DataLoader(trainset, batch_size=8, shuffle=True,
                             collate_fn=batch_graphs)

    # Create model
    model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                                 n_classes=trainset.num_classes)
    logger.info(model)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_graph_classifier(model, data_loader, loss_func=loss_func,
                           optimizer=optimizer, epochs=5)


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    ## Binary Node Classification:
    node_binary_classification(hid_feats=4, out_feats=7, num_heads=2)

    ## Multi-Class Graph Classification:
    graph_multiclass_classification(in_feats=1, hid_feats=4, num_heads=2)


if __name__ == "__main__":
    main()
