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
from torch import nn, stack, Tensor, LongTensor, utils, sigmoid, mean, cat, FloatTensor, BoolTensor
import numpy as np
# import networkx as nx
from json import dumps
# import matplotlib.pyplot as plt
from collections import OrderedDict
from dgl import batch as g_batch, mean_nodes, graph
# from dgl.nn.pytorch.conv import GATConv, GraphConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.gnn_combined import GNN_Combined
from Metrics.metrics import calculate_performance_sk as calculate_performance
from Utils.utils import logit2label
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user, dataset_dir


def train_node_classifier(g: graph, features: Tensor,
                          labels: Tensor, labelled_mask: Tensor,
                          model: GNN_Combined, loss_func,
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


def node_binary_classification(model, hid_feats: int = 4, out_feats: int = 7,
                               num_heads: int = 2) -> None:
    """

    :param model:
    :param hid_feats:
    :param out_feats:
    :param num_heads:
    :return:
    """
    from dgl.data import citation_graph as citegrh

    def load_cora_data():
        data = citegrh.load_cora()
        features = FloatTensor(data.features)
        labels = LongTensor(data.labels)
        mask = BoolTensor(data.train_mask)
        g = graph(data.graph)
        return g, features, labels, mask

    g, features, labels, mask = load_cora_data()

    net = model(in_dim=features.size(1), hidden_dim=hid_feats, out_dim=out_feats, num_heads=num_heads)
    logger.info(net)

    loss_func = F.nll_loss

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    train_node_classifier(g, features, labels, labelled_mask=mask, model=net,
                          loss_func=loss_func, optimizer=optimizer, epochs=5)


def test_node_classifier():
    pass


def train_graph_classifier(model, G, X,
                           data_loader: utils.data.dataloader.DataLoader,
                           loss_func: nn.modules.loss.BCEWithLogitsLoss,
                           optimizer, epochs: int = 5,
                           eval_data_loader: utils.data.dataloader.DataLoader = None):
    logger.info("Started training...")
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(data_loader):
            ## Store emb in a separate file as self_loop removes emb info:
            emb = graph_batch.ndata['emb']
            # graph_batch = dgl.add_self_loop(graph_batch)
            prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X)
            loss = loss_func(prediction, label)
            logger.info(f"Iteration {iter}, loss: {loss.detach().item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            preds.append(prediction.detach())
            trues.append(label.detach())
        epoch_loss /= (iter + 1)
        losses, test_output = test_graph_classifier(
            model, G, X, loss_func=loss_func, data_loader=eval_data_loader)
        logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, Eval loss {losses},"
                    f" Macro F1 {test_output['result']['f1']['macro'].item()}")
        logger.info(dumps(test_output['result'], indent=4))
        train_epoch_losses.append(epoch_loss)
        preds = cat(preds)

        ## Converting raw scores to probabilities using Sigmoid:
        preds = sigmoid(preds)

        ## Converting probabilities to class labels:
        preds = logit2label(preds.detach(), cls_thresh=0.5)
        trues = cat(trues)
        result_dict = calculate_performance(trues, preds)
        # logger.info(dumps(result_dict, indent=4))
        train_epoch_dict[epoch] = {
            'preds':  preds,
            'trues':  trues,
            'result': result_dict
        }
        # logger.info(f'Epoch {epoch} result: \n{result_dict}')

    return train_epoch_losses, train_epoch_dict


def test_graph_classifier(model: GNN_Combined, G, X, loss_func,
                          data_loader: utils.data.dataloader.DataLoader):
    model.eval()
    preds = []
    trues = []
    losses = []
    for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(data_loader):
        ## Store emb in a separate file as self_loop removes emb info:
        emb = graph_batch.ndata['emb']
        # graph_batch = dgl.add_self_loop(graph_batch)
        prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X)
        loss = loss_func(prediction, label)
        preds.append(prediction.detach())
        trues.append(label.detach())
        losses.append(loss.detach())
    losses = mean(stack(losses))
    preds = cat(preds)

    ## Converting raw scores to probabilities using Sigmoid:
    preds = sigmoid(preds)

    ## Converting probabilities to class labels:
    preds = logit2label(preds.detach(), cls_thresh=0.5)
    trues = cat(trues)
    result_dict = calculate_performance(trues, preds)
    test_output = {
        'preds':  preds,
        'trues':  trues,
        'result': result_dict
    }
    # logger.info(dumps(result_dict, indent=4))

    return losses, test_output


# def predict_with_label(model, iterator, criterion=None, metric=True):
#     """ Predicts and calculates performance. Labels mandatory
#
#     Args:
#         model:
#         iterator:
#         criterion:
#
#     Returns:
#     :param metric:
#
#     """
#     # initialize every epoch
#     epoch_loss = 0
#
#     if criterion is None:
#         criterion = torch.nn.BCEWithLogitsLoss()
#
#     preds_trues = {'preds': [], 'trues': [], 'ids': [], 'losses': [], 'results': []}
#
#     # deactivating dropout layers
#     model.eval()
#
#     # deactivates autograd
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             # retrieve text and no. of words
#             text, text_lengths = batch.text
#
#             # convert to 1d tensor
#             predictions = model(text, text_lengths).squeeze()
#
#             # compute loss and accuracy
#             batch_labels = torchtext_batch2multilabel(batch)
#             preds_trues['preds'].append(predictions)
#             preds_trues['trues'].append(batch_labels)
#             preds_trues['ids'].append(batch.ids)
#             loss = criterion(predictions, batch_labels)
#
#             # keep track of loss and accuracy
#             epoch_loss += loss.item()
#             preds_trues['losses'].append(epoch_loss)
#             # epoch_acc += acc.item()
#             # epoch_acc += acc["accuracy"]["unnormalize"]
#         if metric:
#             ## Converting raw scores to probabilities using Sigmoid:
#             preds = torch.sigmoid(predictions)
#
#             ## Converting probabilities to class labels:
#             preds = logit2label(preds.detach(), cls_thresh=0.5)
#             trues = torch.cat(preds_trues['trues'])
#             result_dict = calculate_performance(trues, preds)
#
#         preds_trues['preds'] = torch.cat(preds_trues['preds'])
#         preds_trues['trues'] = torch.cat(preds_trues['trues'])
#         preds_trues['ids'] = torch.cat(preds_trues['ids'])
#         preds_trues['losses'] = torch.cat(preds_trues['losses'])
#
#     return epoch_loss / len(iterator), preds_trues


def graph_multilabel_classification(
        G, X, train_dataloader, test_dataloader, num_tokens: int, in_feats: int = 100,
        hid_feats: int = 50, num_heads: int = 2, epochs=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"]):
    # train_dataloader, test_dataloader = dataloaders
    model = GNN_Combined(num_tokens, in_dim=in_feats, hidden_dim=hid_feats,
                         num_heads=num_heads, out_dim=hid_feats,
                         num_classes=train_dataloader.dataset.num_labels)
    logger.info(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses, train_epochs_output_dict = train_graph_classifier(
        model, G, X, train_dataloader, loss_func=loss_func,
        optimizer=optimizer, epochs=epochs, eval_data_loader=test_dataloader)

    losses, test_output = test_graph_classifier(
        model, G, X, loss_func=loss_func, data_loader=test_dataloader)
    logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict, test_output


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    graph_multilabel_classification(in_feats=1, hid_feats=4, num_heads=2)


if __name__ == "__main__":
    main()
