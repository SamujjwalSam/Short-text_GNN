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

import timeit
from torch import nn, stack, utils, sigmoid, mean, cat, device, cuda, save
from os import environ
from json import dumps
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.glen_classifier import GLEN_Classifier
from Metrics.metrics import calculate_performance_sk as calculate_performance,\
    calculate_performance_bin_sk
from Utils.utils import logit2label, count_parameters
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user

device = device('cuda' if cuda.is_available() else 'cpu')
if cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def train_graph_classifier(model, G, X,
                           data_loader: utils.data.dataloader.DataLoader,
                           loss_func: nn.modules.loss.BCEWithLogitsLoss,
                           optimizer, epochs: int = 5,
                           eval_data_loader: utils.data.dataloader.DataLoader = None,
                           test_data_loader: utils.data.dataloader.DataLoader = None,
                           n_classes=cfg['data']['num_classes']):
    logger.info("Started training...")
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        start_time = timeit.default_timer()
        for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(data_loader):
            ## Store emb in a separate file as self_loop removes emb info:
            emb = graph_batch.ndata['emb']
            # graph_batch = dgl.add_self_loop(graph_batch)
            if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
                graph_batch = graph_batch.to(device)
                emb = emb.to(device)
                # local_ids = local_ids.to(device)
                # node_counts = node_counts.to(device)
                # global_ids = global_ids.to(device)
                # G = G.to(device)
                # X = X.to(device)
            prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X)
            # if epoch == 30:
            #     from evaluations import get_freq_disjoint_token_vecs, plot
            #     glove_vecs = get_freq_disjoint_token_vecs(S_vocab, T_vocab, X)
            #     plot(glove_vecs)
            if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
                prediction = prediction.to(device)
                label = label.to(device)
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(1)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_count = label.shape[0]
            epoch_loss += loss.detach().item()
            if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
                preds.append(prediction.detach().cpu())
                trues.append(label.detach().cpu())
                # losses.append(loss.detach().cpu())
            else:
                preds.append(prediction.detach())
                trues.append(label.detach())
                # losses.append(loss.detach())
                # preds.append(prediction.detach())
                # trues.append(label.detach())

        epoch_loss /= (iter + 1)
        train_time = timeit.default_timer() - start_time
        logger.info(f"Epoch {epoch}, time: {train_time / 60} mins, loss: {epoch_loss}")
        val_losses, val_output = eval_graph_classifier(
            model, G, X, loss_func=loss_func, data_loader=eval_data_loader)
        logger.info(f'val_output: \n{dumps(val_output["result"], indent=4)}')
        test_losses, test_output = eval_graph_classifier(
            model, G, X, loss_func=loss_func, data_loader=test_data_loader)
        logger.info(f'test_output: \n{dumps(test_output["result"], indent=4)}')
        logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, val loss "
                    f"{val_losses}, test loss {test_losses}, Val Weighted F1 "
                    f"{val_output['result']['f1']['weighted'].item()} Test Weighted F1"
                    f" {test_output['result']['f1']['weighted'].item()}")
        # logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, val loss "
        #             f"{val_losses}, Val Weighted F1 {val_output['result']['f1']['weighted'].item()}")
        train_epoch_losses.append(epoch_loss)
        preds = cat(preds)

        ## Converting raw scores to probabilities using Sigmoid:
        preds = sigmoid(preds)

        ## Converting probabilities to class labels:
        preds = logit2label(preds.detach(), cls_thresh=0.5)
        trues = cat(trues)
        if n_classes == 1:
            result_dict = calculate_performance_bin_sk(trues, preds)
        else:
            result_dict = calculate_performance(trues, preds)
        # logger.info(dumps(result_dict, indent=4))
        train_epoch_dict[epoch] = {
            'preds':  preds,
            'trues':  trues,
            'result': result_dict
        }
        # logger.info(f'Epoch {epoch} result: \n{result_dict}')

    return train_epoch_losses, train_epoch_dict


def eval_graph_classifier(model: GLEN_Classifier, G, X, loss_func,
                          data_loader: utils.data.dataloader.DataLoader,
                          n_classes=cfg['data']['num_classes'],
                          save_gcn_embs=False):
    model.eval()
    preds = []
    trues = []
    losses = []
    start_time = timeit.default_timer()
    for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(data_loader):
        ## Store emb in a separate file as self_loop removes emb info:
        emb = graph_batch.ndata['emb']
        # graph_batch = dgl.add_self_loop(graph_batch)
        if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
            graph_batch = graph_batch.to(device)
            emb = emb.to(device)
            # local_ids = local_ids.to(device)
            # node_counts = node_counts.to(device)
            # global_ids = global_ids.to(device)
            G = G.to(device)
            X = X.to(device)
        if save_gcn_embs:
            save(X, 'X_glove.pt')
        prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X, save_gcn_embs)
        # test_count = label.shape[0]
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
        if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
            prediction = prediction.to(device)
            label = label.to(device)
        loss = loss_func(prediction, label)
        if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
            preds.append(prediction.detach().cpu())
            trues.append(label.detach().cpu())
            losses.append(loss.detach().cpu())
        else:
            preds.append(prediction.detach())
            trues.append(label.detach())
            losses.append(loss.detach())
    test_time = timeit.default_timer() - start_time
    logger.info(f"Total test time: [{test_time / 60} mins]")
    losses = mean(stack(losses))
    preds = cat(preds)

    ## Converting raw scores to probabilities using Sigmoid:
    preds = sigmoid(preds)

    ## Converting probabilities to class labels:
    preds = logit2label(preds.detach().cpu(), cls_thresh=0.5)
    trues = cat(trues)
    if n_classes == 1:
        result_dict = calculate_performance_bin_sk(trues, preds)
    else:
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


def GLEN_trainer(
        G, X, train_dataloader, val_dataloader, test_dataloader, num_tokens: int, in_feats: int = 100,
        hid_feats: int = 50, num_heads: int = 2, epochs=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        n_classes=cfg['data']['num_classes'], state=None):
    # train_dataloader, test_dataloader = dataloaders
    model = GLEN_Classifier(
        num_tokens, in_dim=in_feats, hidden_dim=hid_feats, num_heads=num_heads,
        out_dim=hid_feats, num_classes=train_dataloader.dataset.num_labels,
        state=state)
    logger.info(model)
    count_parameters(model)
    if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
        model.to(device)
        G = G.to(device)
        X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Load optimizer state and params:
    # if state is not None:
    #     optimizer.load_state_dict(state['optimizer'])

    epoch_losses, train_epochs_output_dict = train_graph_classifier(
        model, G, X, train_dataloader, loss_func=loss_func,
        optimizer=optimizer, epochs=epochs, eval_data_loader=val_dataloader,
        test_data_loader=test_dataloader)

    start_time = timeit.default_timer()
    losses, test_output = eval_graph_classifier(
        model, G, X, loss_func=loss_func, data_loader=test_dataloader, save_gcn_embs=True)
    test_time = timeit.default_timer() - start_time
    test_count = test_dataloader.dataset.__len__()
    logger.info(f"Total inference time for [{test_count}] examples: [{test_time} sec]"
                f"\nPer example: [{test_time / test_count} sec]")
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
    GLEN_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
