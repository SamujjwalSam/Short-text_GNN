# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict

from Logger.logger import logger
from Utils.utils import save_model, load_model


# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 4


def torchtext_batch2multilabel(batch, label_cols=None):
    """ Returns labels for a TorchText batch.

    Args:
        batch:
        label_cols:

    Returns:

    """
    if label_cols is None:
        label_cols = [str(cls) for cls in range(n_classes)]
    return torch.cat([getattr(batch, feat).unsqueeze(1) for feat in label_cols],
                     dim=1).float()


def training(model, iterator, optimizer, criterion):
    """ Model training function.

    Args:
        model:
        iterator:
        optimizer:
        criterion:

    Returns:

    """
    # initialize every epoch
    epoch_loss = 0

    stores = {'preds': [], 'trues': [], 'ids': []}

    # set the model in training phase
    model.train()

    for i, batch in enumerate(iterator):
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        batch_labels = torchtext_batch2multilabel(batch)
        stores['preds'].append(predictions)
        stores['trues'].append(batch_labels)
        stores['ids'].append(batch.ids)
        loss = criterion(predictions, batch_labels)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        # logger.info(f"Train batch [{i}] loss: [{loss.item()}]")

    stores['preds'] = torch.cat(stores['preds'])
    stores['trues'] = torch.cat(stores['trues'])
    stores['ids'] = torch.cat(stores['ids'])

    return epoch_loss / len(iterator), stores  # , epoch_acc / len(iterator)


def predict_with_label(model, iterator, criterion=None):
    """ Predicts and calculates performance. Labels mandatory

    Args:
        model:
        iterator:
        criterion:

    Returns:

    """
    # initialize every epoch
    epoch_loss = 0

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    preds_trues = {'preds': [], 'trues': [], 'ids': []}

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            batch_labels = torchtext_batch2multilabel(batch)
            preds_trues['preds'].append(predictions)
            preds_trues['trues'].append(batch_labels)
            preds_trues['ids'].append(batch.ids)
            loss = criterion(predictions, batch_labels)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            # epoch_acc += acc.item()
            # epoch_acc += acc["accuracy"]["unnormalize"]

        preds_trues['preds'] = torch.cat(preds_trues['preds'])
        preds_trues['trues'] = torch.cat(preds_trues['trues'])
        preds_trues['ids'] = torch.cat(preds_trues['ids'])

    return epoch_loss / len(iterator), preds_trues


def trainer(model, train_iterator, val_iterator, N_EPOCHS=5, optimizer=None,
            criterion=None, best_val_loss=float('inf'), lr=0.001):
    """ Trains the model.

    :param model:
    :param train_iterator:
    :param val_iterator:
    :param N_EPOCHS:
    :param optimizer:
    :param criterion:
    :param best_val_loss:
    :return:
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss().to(device)

    # push to cuda if available
    model = model.to(device)
    criterion = criterion.to(device)

    sigmoid = nn.Sigmoid()
    val_preds_trues_all = OrderedDict()
    val_preds_trues_best = None
    model_best = None
    record_train_losses = []
    record_val_losses = []
    ## TODO: Run epochs till val_loss decreases
    for epoch in range(N_EPOCHS):
        logger.info("=" * 10)
        logger.info("Epoch: [{}]".format(epoch))
        # train the model
        train_loss, train_preds = training(model, train_iterator, optimizer,
                                           criterion)

        # evaluate the model
        val_loss, val_preds_trues = predict_with_label(model, val_iterator,
                                                       criterion)
        val_preds_trues['preds'] = sigmoid(val_preds_trues['preds'])
        # val_preds_trues_all[epoch] = val_preds_trues

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_preds_trues_best = val_preds_trues
            model_best = model
            # torch.save(model.state_dict(), 'saved_weights.pt')
            save_model(model)
            # save_model(saved_model_name='tweet_bilstm_'+str(epoch))

        logger.info(f'\tTrain Loss: {train_loss:.6f}')
        record_train_losses.append(train_loss)
        logger.info(f'\t Val. Loss: {val_loss:.6f}')
        record_val_losses.append(val_loss)

    losses = {
        "train": record_train_losses,
        "val":   record_val_losses
    }

    return model_best, val_preds_trues_best, val_preds_trues_all, losses
