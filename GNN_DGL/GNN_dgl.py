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
import dgl
import dgl.function as fn
# import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch as th
from torch.nn import init
import torch
import numpy as np
# import networkx as nx
from json import dumps
# import matplotlib.pyplot as plt
from collections import OrderedDict
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl import DGLGraph, batch as g_batch, mean_nodes
from dgl.nn.pytorch.conv import GATConv, GraphConv
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Metrics.metrics import calculate_performance_sk as calculate_performance
from Utils.utils import logit2label
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user, dataset_dir


# class GraphConv(nn.Module):
#     r"""
#     Parameters
#     ----------
#     in_feats : int
#         Input feature size.
#     out_feats : int
#         Output feature size.
#     bias : bool, optional
#         If True, adds a learnable bias to the output. Default: ``True``.
#     activation: callable activation function/layer or None, optional
#         If not None, applies an activation function to the updated node features.
#         Default: ``None``.
#
#     Attributes
#     ----------
#     weight : torch.Tensor
#         The learnable weight tensor.
#     bias : torch.Tensor
#         The learnable bias tensor.
#     """
#
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  bias=True,
#                  activation=None):
#         super(GraphConv, self).__init__()
#
#         self._in_feats = in_feats
#         self._out_feats = out_feats
#         self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
#
#         if bias:
#             self.bias = nn.Parameter(th.Tensor(out_feats))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#         self._activation = activation
#
#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         if self.weight is not None:
#             init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             init.zeros_(self.bias)
#
#     def forward(self, graph, feat, eweight):
#         r"""Compute graph convolution.
#
#         Notes
#         -----
#         * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
#           dimensions, :math:`N` is the number of nodes.
#         * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
#           the same shape as the input.
#
#         Parameters
#         ----------
#         graph : DGLGraph
#             The graph.
#         feat : torch.Tensor or pair of torch.Tensor
#             If a torch.Tensor is given, it represents the input feature of shape
#             :math:`(N, D_{in})`
#             where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
#             If a pair of torch.Tensor is given, the pair must contain two tensors of shape
#             :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
#             Note that in the special case of graph convolutional networks, if a pair of
#             tensors is given, the latter element will not participate in computation.
#         eweight : torch.Tensor of shape (E, 1)
#             Values associated with the edges in the adjacency matrix.
#
#         Returns
#         -------
#         torch.Tensor
#             The output feature
#         """
#         with graph.local_scope():
#             feat_src, feat_dst = expand_as_pair(feat, graph)
#
#             if self._in_feats > self._out_feats:
#                 # mult W first to reduce the feature size for aggregation.
#                 feat_src = th.matmul(feat_src, self.weight)
#                 graph.srcdata['h'] = feat_src
#                 graph.edata['w'] = eweight
#                 graph.update_all(fn.u_mul_e('h', 'w', 'm'),
#                                  fn.sum('m', 'h'))
#                 rst = graph.dstdata['h']
#             else:
#                 # aggregate first then mult W
#                 graph.srcdata['h'] = feat_src
#                 graph.edata['w'] = eweight
#                 graph.update_all(fn.u_mul_e('h', 'w', 'm'),
#                                  fn.sum('m', 'h'))
#                 rst = graph.dstdata['h']
#                 rst = th.matmul(rst, self.weight)
#
#             if self.bias is not None:
#                 rst = rst + self.bias
#
#             if self._activation is not None:
#                 rst = self._activation(rst)
#
#             return rst
#
#     def extra_repr(self):
#         """Set the extra representation of the module,
#         which will come into effect when printing the model.
#         """
#         summary = 'in={_in_feats}, out={_out_feats}'
#         summary += ', normalization={_norm}'
#         if '_activation' in self.__dict__:
#             summary += ', activation={_activation}'
#         return summary.format(**self.__dict__)


## GCN with precomputed normalized adjacency matrix and edge features:
# The norm can be computed as follows
def normalize_adj(g):
    in_degs = g.in_degrees().float()
    in_norm = th.pow(in_degs, -0.5).unsqueeze(-1)
    out_degs = g.out_degrees().float()
    out_norm = th.pow(out_degs, -0.5).unsqueeze(-1)
    g.ndata['in_norm'] = in_norm
    g.ndata['out_norm'] = out_norm
    g.apply_edges(fn.u_mul_v('in_norm', 'out_norm', 'eweight'))

    return g


class GNN_Layer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GNN_Layer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 2)))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class GCN(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNN_Layer(ndim_in, edim, 50, activation))
        self.layers.append(GNN_Layer(50, edim, 25, activation))
        self.layers.append(GNN_Layer(25, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


if __name__ == '__main__':
    model = GCN(3, 1, 3, F.relu, 0.5)
    g = dgl.DGLGraph([[0, 2], [2, 3]])
    nfeats = torch.randn((g.number_of_nodes(), 3, 3))
    efeats = torch.randn((g.number_of_edges(), 3, 3))
    model(g, nfeats, efeats)


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
        # Be aware that the input dimension is hid_dim*num_heads since
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
                           dataloader: torch.utils.data.dataloader.DataLoader,
                           loss_func: torch.nn.modules.loss.BCEWithLogitsLoss,
                           optimizer, epochs: int = 5,
                           eval_dataloader: torch.utils.data.dataloader.DataLoader = None):
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        for iter, (graph_batch, label) in enumerate(dataloader):
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
            model, loss_func=loss_func, dataloader=eval_dataloader)
        logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, Eval loss {losses},"
                    f" Weighted F1 {test_output['result']['f1']['weighted'].item()}")
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
                          dataloader: torch.utils.data.dataloader.DataLoader):
    model.eval()
    preds = []
    trues = []
    losses = []
    for iter, (graph_batch, label) in enumerate(dataloader):
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

    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["model"]["optimizer"]["lr"])

    epoch_losses, train_epochs_output_dict = train_graph_classifier(
        model, gdh.train_dataloader(), loss_func=loss_func,
        optimizer=optimizer, epochs=epochs,
        eval_dataloader=gdh.test_dataloader())

    losses, test_output = test_graph_classifier(model, loss_func=loss_func,
                                                dataloader=gdh.test_dataloader())
    logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict, test_output


def graph_multiclass_classification(in_feats: int = 1, hid_feats: int = 4, num_heads: int = 2) -> None:
    from dgl.data import MiniGCDataset

    # Create training and test sets.
    trainset = MiniGCDataset(320, 10, 20)
    testset = MiniGCDataset(80, 10, 20)

    # # Use PyTorch's DataLoader and the collate function defined before.
    dataloader = DataLoader(trainset, batch_size=8, shuffle=True,
                            collate_fn=batch_graphs)

    # Create model
    model = GAT_Graph_Classifier(in_feats, hid_feats, num_heads=num_heads,
                                 out_dim=trainset.num_classes)
    logger.info(model)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses, epoch_predictions_dict = train_graph_classifier(
        model, dataloader, loss_func=loss_func, optimizer=optimizer, epochs=5)


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
