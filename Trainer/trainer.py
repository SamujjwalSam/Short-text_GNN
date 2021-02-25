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
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from torch import cuda, load, sigmoid, cat, optim, save, device, no_grad
# import torch.optim as optim
import torch.nn as nn
from os.path import join
from collections import OrderedDict

from Logger.logger import logger
from Utils.utils import save_model, load_model, logit2label
from Metrics.metrics import calculate_performance_sk as calculate_performance
from config import configuration as cfg, platform as plat, username as user, dataset_dir

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'])
    cuda.set_device(cfg['cuda']['cuda_devices'])


def torchtext_batch2multilabel(batch, label_cols=None):
    """ Returns labels for a TorchText batch.

    Args:
        batch:
        label_cols:

    Returns:

    """
    if label_cols is None:
        label_cols = [str(cls) for cls in range(n_classes)]
    return cat([getattr(batch, feat).unsqueeze(1) for feat in label_cols],
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

    output = {'preds': [], 'trues': [], 'ids': []}

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
        output['preds'].append(predictions)
        output['trues'].append(batch_labels)
        output['ids'].append(batch.ids)
        loss = criterion(predictions, batch_labels)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        # logger.info(f"Train batch [{i}] loss: [{loss.item()}]")

    output['preds'] = cat(output['preds'])
    output['trues'] = cat(output['trues'])
    output['ids'] = cat(output['ids'])

    return epoch_loss / len(iterator), output  # , epoch_acc / len(iterator)


def predict_with_label(model, iterator, criterion=None, metric=True):
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

    preds_trues = {'preds': [], 'trues': [], 'ids': [], 'losses': [], 'results': []}

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with no_grad():
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
            preds_trues['losses'].append(epoch_loss)
            # epoch_acc += acc.item()
            # epoch_acc += acc["accuracy"]["unnormalize"]
        if metric:
            ## Converting raw scores to probabilities using Sigmoid:
            preds = sigmoid(predictions)

            ## Converting probabilities to class labels:
            preds = logit2label(preds.detach(), cls_thresh=0.5)
            trues = cat(preds_trues['trues'])
            result_dict = calculate_performance(trues, preds)

        preds_trues['preds'] = cat(preds_trues['preds'])
        preds_trues['trues'] = cat(preds_trues['trues'])
        preds_trues['ids'] = cat(preds_trues['ids'])
        preds_trues['losses'] = cat(preds_trues['losses'])

    return epoch_loss / len(iterator), preds_trues


def trainer(model, train_iterator, val_iterator, N_EPOCHS=5, optimizer=None,
            criterion=None, best_val_loss=float('inf'), lr=cfg["model"]["optimizer"]["lr"]):
    """ Trains the model.

    :param lr:
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
    ## TODO: Run epoch till val_loss decreases
    for epoch in range(N_EPOCHS):
        logger.info("=" * 10)
        logger.info("Epoch: [{}]".format(epoch))
        # train the model
        train_loss, train_preds = training(model, train_iterator, optimizer, criterion)

        # evaluate the model
        val_loss, val_preds_trues = predict_with_label(model, val_iterator, criterion)
        val_preds_trues['preds'] = sigmoid(val_preds_trues['preds'])
        record_train_losses.append(train_loss)
        record_val_losses.append(val_loss)

        # logger.info(f"Epoch {epoch}, Train loss {train_loss}, Eval loss {val_loss},"
        #             f" Weighted F1 {test_output['result']['f1']['weighted'].item()}")
        # val_preds_trues_all[epoch] = val_preds_trues

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_preds_trues_best = val_preds_trues
            model_best = model
            # torch.save(model.state_dict(), 'saved_weights.pt')
            save_model(model)
            # save_model(saved_model_name='tweet_bilstm_'+str(epoch))

    losses = {
        "train": record_train_losses,
        "val":   record_val_losses
    }

    return model_best, val_preds_trues_best, val_preds_trues_all, losses


def save_model(model, saved_model_dir=dataset_dir, saved_model_name='model'):
    try:
        save(model.state_dict(), join(saved_model_dir, saved_model_name))
    except Exception as e:
        logger.fatal(
            "Could not save model at [{}] due to Error: [{}]".format(join(
                saved_model_dir, saved_model_name), e))
        return False
    return True


def load_model(model, saved_model_dir, saved_model_name='model'):
    try:
        model.load_state_dict(load(join(saved_model_dir, saved_model_name)))
    except Exception as e:
        logger.fatal(
            "Could not load model from [{}] due to Error: [{}]".format(join(
                saved_model_dir, saved_model_name), e))
        return False
    model.eval()
    return model


def get_optimizer(model, lr=cfg["model"]["optimizer"]["lr"],
                  optimizer_type=cfg["model"]["optimizer"]["optimizer_type"],
                  weight_decay=cfg["model"]["optimizer"]["weight_decay"],
                  rho=cfg["model"]["optimizer"]["rho"],
                  lr_decay=cfg["model"]["optimizer"]["lr_decay"],
                  momentum=cfg["model"]["optimizer"]["momentum"],
                  dampening=cfg["model"]["optimizer"]["dampening"],
                  alpha=cfg["model"]["optimizer"]["alpha"],
                  centered=cfg["model"]["optimizer"]["centered"]):
    """Setup optimizer_type"""
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              dampening=dampening, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=rho,
                                   weight_decay=weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=0.9,
                                  centered=centered, weight_decay=weight_decay)
    else:
        raise Exception(f'Optimizer not supported: [{optimizer_type}]')
    return optimizer
