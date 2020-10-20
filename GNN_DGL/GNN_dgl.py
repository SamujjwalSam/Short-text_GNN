# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Code related to applying GNN from DGL library
__description__ : node and graph classification written in DGL library
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
# import dgl
import torch
import numpy as np
# import networkx as nx
from json import dumps
# import matplotlib.pyplot as plt
from collections import OrderedDict
from dgl import DGLGraph, batch as g_batch, mean_nodes
from dgl.nn.pytorch.conv import GATConv, GraphConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Metrics.metrics import calculate_performance_sk as calculate_performance
from Utils.utils import logit2label
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user


class GCN_Node_Classifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN_Node_Classifier, self).__init__()
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


class GAT_Node_Classifier(torch.nn.Module):
    """ Graph Attention Network model

    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int) -> None:
        super(GAT_Node_Classifier, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g: DGLGraph, emb: torch.Tensor) -> torch.Tensor:
        if emb is None:
            # Use node degree as the initial node feature. For undirected graphs,
            # the in-degree is the same as the out_degree.
            # emb = g.in_degrees().view(-1, 1).float()
            emb = g.ndata['emb']
        emb = self.layer1(g, emb)
        ## Concatenating multiple head embeddings
        emb = emb.view(-1, emb.size(1) * emb.size(2))
        emb = F.elu(emb)
        emb = self.layer2(g, emb).squeeze()
        return emb


def train_node_classifier(g: DGLGraph, features: torch.Tensor,
                          labels: torch.Tensor, labelled_mask: torch.Tensor,
                          model: GAT_Node_Classifier, loss_func,
                          optimizer, epochs: int = 5) -> None:
    """

    :param g:
    :param features:
    :param labels:
    :param labelled_mask:
    :param model:
    :param loss_func:
    :param optimizer:
    :param epochs:
    """
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


def node_binary_classification(hid_feats: int = 4, out_feats: int = 7,
                               num_heads: int = 2) -> None:
    """

    :param hid_feats:
    :param out_feats:
    :param num_heads:
    :return:
    """
    from dgl.data import citation_graph as citegrh

    def load_cora_data():
        data = citegrh.load_cora()
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        mask = torch.BoolTensor(data.train_mask)
        g = DGLGraph(data.graph)
        return g, features, labels, mask

    g, features, labels, mask = load_cora_data()

    net = GAT_Node_Classifier(in_dim=features.size(1), hidden_dim=hid_feats,
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
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, out_dim: int) -> None:
        super(GAT_Graph_Classifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.classify = torch.nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, g: DGLGraph, emb: torch.Tensor = None) -> torch.Tensor:
        if emb is None:
            # Use node degree as the initial node feature. For undirected graphs,
            # the in-degree is the same as the out_degree.
            # emb = g.in_degrees().view(-1, 1).float()
            emb = g.ndata['emb']

        # Perform graph convolution and activation function.
        emb = F.relu(self.conv1(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        emb = F.relu(self.conv2(g, emb))
        emb = emb.view(-1, emb.size(1) * emb.size(2)).float()
        g.ndata['emb'] = emb

        # Calculate graph representation by averaging all node representations.
        hg = mean_nodes(g, 'emb')
        # hg = readout_nodes(g, 'emb')
        return self.classify(hg)


def train_graph_classifier(model: GAT_Graph_Classifier,
                           data_loader: torch.utils.data.dataloader.DataLoader,
                           loss_func: torch.nn.modules.loss.BCEWithLogitsLoss,
                           optimizer, epochs: int = 5,
                           eval_data_loader: torch.utils.data.dataloader.DataLoader = None, ):
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        for iter, (graph_batch, label) in enumerate(data_loader):
            ## Store emb in a separate file as self_loop removes emb info:
            emb = graph_batch.ndata['emb']
            # graph_batch = dgl.add_self_loop(graph_batch)
            prediction = model(graph_batch, emb)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            preds.append(prediction.detach())
            trues.append(label.detach())
        epoch_loss /= (iter + 1)
        losses, test_output = test_graph_classifier(
            model, loss_func=loss_func, data_loader=eval_data_loader)
        logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, Eval loss {losses},"
                    f" Macro F1 {test_output['result']['f1']['macro'].item()}")
        # logger.info(dumps(test_output['result'], indent=4))
        train_epoch_losses.append(epoch_loss)
        preds = torch.cat(preds)

        ## Converting raw scores to probabilities using Sigmoid:
        preds = torch.sigmoid(preds)

        ## Converting probabilities to class labels:
        preds = logit2label(preds.detach(), cls_thresh=0.5)
        trues = torch.cat(trues)
        result_dict = calculate_performance(trues, preds)
        # logger.info(dumps(result_dict, indent=4))
        train_epoch_dict[epoch] = {
            'preds':  preds,
            'trues':  trues,
            'result': result_dict
        }
        # logger.info(f'Epoch {epoch} result: \n{result_dict}')

    return train_epoch_losses, train_epoch_dict


def test_graph_classifier(model: GAT_Graph_Classifier, loss_func,
                          data_loader: torch.utils.data.dataloader.DataLoader):
    model.eval()
    preds = []
    trues = []
    losses = []
    for iter, (graph_batch, label) in enumerate(data_loader):
        ## Store emb in a separate file as self_loop removes emb info:
        emb = graph_batch.ndata['emb']
        # graph_batch = dgl.add_self_loop(graph_batch)
        prediction = model(graph_batch, emb)
        loss = loss_func(prediction, label)
        preds.append(prediction.detach())
        trues.append(label.detach())
        losses.append(loss.detach())
    losses = torch.mean(torch.stack(losses))
    preds = torch.cat(preds)

    ## Converting raw scores to probabilities using Sigmoid:
    preds = torch.sigmoid(preds)

    ## Converting probabilities to class labels:
    preds = logit2label(preds.detach(), cls_thresh=0.5)
    trues = torch.cat(trues)
    result_dict = calculate_performance(trues, preds)
    test_output = {
        'preds':  preds,
        'trues':  trues,
        'result': result_dict
    }
    # logger.info(dumps(result_dict, indent=4))

    return losses, test_output


def graph_multilabel_classification(
        gdh, in_feats: int = 100, hid_feats: int = 50, num_heads: int = 2,
        epochs=cfg['training']['num_epoch']):
    model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                                 out_dim=gdh.num_classes)
    logger.info(model)

    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["model"]["optimizer"]["lr"])

    epoch_losses, train_epochs_output_dict = train_graph_classifier(
        model, gdh.train_dataloader(), loss_func=loss_func,
        optimizer=optimizer, epochs=epochs,
        eval_data_loader=gdh.test_dataloader())

    losses, test_output = test_graph_classifier(model, loss_func=loss_func,
                                                data_loader=gdh.test_dataloader())
    logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict, test_output


def graph_multiclass_classification(in_feats: int = 1, hid_feats: int = 4, num_heads: int = 2) -> None:
    from dgl.data import MiniGCDataset

    # Create training and test sets.
    trainset = MiniGCDataset(320, 10, 20)
    testset = MiniGCDataset(80, 10, 20)

    # # Use PyTorch's DataLoader and the collate function defined before.
    data_loader = DataLoader(trainset, batch_size=8, shuffle=True,
                             collate_fn=batch_graphs)

    # Create model
    model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                                 out_dim=trainset.num_classes)
    logger.info(model)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses, epoch_predictions_dict = train_graph_classifier(
        model, data_loader, loss_func=loss_func, optimizer=optimizer, epochs=5)


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
