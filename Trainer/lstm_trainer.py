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
from torch import nn, stack, utils, sigmoid, mean, cat, cuda, save, load
from os import environ, mkdir
from os.path import join, exists
from json import dumps
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.bilstm_classifiers import BiLSTM_Classifier, BiLSTM_Emb_Classifier
from Metrics.metrics import calculate_performance_sk as calculate_performance,\
    calculate_performance_bin_sk
from Utils.utils import logit2label, count_parameters, save_model_state, load_model_state
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user, device

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'])
    cuda.set_device(cfg['cuda']['cuda_devices'])


def train_lstm_classifier(
        model: BiLSTM_Emb_Classifier, dataloader: utils.data.dataloader.DataLoader,
        loss_func: nn.modules.loss.BCEWithLogitsLoss, optimizer,
        epoch: int = cfg['training']['num_epoch'],
        eval_dataloader: utils.data.dataloader.DataLoader = None,
        test_dataloader: utils.data.dataloader.DataLoader = None,
        n_classes=cfg['data']['num_classes'], model_name='Glove'):
    logger.info(f"Started training for {epoch} epoch: ")
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(1, epoch+1):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        start_time = timeit.default_timer()
        for iter, batch in enumerate(dataloader):
            text, text_lengths = batch.text
            ## Get label based on number of classes:
            if cfg['data']['class_names'] == 1:
                label = batch.__getattribute__('0').unsqueeze(1)
            else:
                label = stack([batch.__getattribute__(cls) for cls in cfg['data']['class_names']]).T
            prediction = model(text, text_lengths).squeeze()
            if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
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
            if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
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

        ## Don't save model if name starts with 'no'
        # if not model_name.startswith('WSCP'):
        # save_model_state(model, model_name + '_epoch' + str(epoch), optimizer=optimizer)

        val_losses, val_output = eval_lstm_classifier(
            model, loss_func=loss_func, dataloader=eval_dataloader)
        # logger.info(f'val_output: \n{dumps(val_output["result"], indent=4)}')
        test_losses, test_output = eval_lstm_classifier(
            model, loss_func=loss_func, dataloader=test_dataloader)
        logger.info(f'test_output: \n{dumps(test_output["result"], indent=4)}')
        logger.info(f"Epoch {epoch}, "
                    # f"Train loss {epoch_loss}, val loss {val_losses}, test loss {test_losses}, "
                    f"Val W-F1 {val_output['result']['f1']['weighted'].item():4.4} "
                    f"Test W-F1 {test_output['result']['f1']['weighted'].item():4.4} "
                    f"Model {model_name}")
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


def eval_lstm_classifier(model: BiLSTM_Emb_Classifier, loss_func,
                         dataloader: utils.data.dataloader.DataLoader,
                         n_classes=cfg['data']['num_classes']):
    # if use_saved:
    #     model = load_model_state(model, epoch)

    model.eval()
    preds = []
    trues = []
    losses = []
    start_time = timeit.default_timer()
    for iter, batch in enumerate(dataloader):
        text, text_lengths = batch.text
        ## Get label based on number of classes:
        if cfg['data']['class_names'] == 1:
            label = batch.__getattribute__('0').unsqueeze(1)
        else:
            label = stack([batch.__getattribute__(cls) for cls in cfg['data']['class_names']]).T
        prediction = model(text, text_lengths)
        # test_count = label.shape[0]
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
        if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
            prediction = prediction.to(device)
            label = label.to(device)
        loss = loss_func(prediction, label)
        if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
            preds.append(prediction.detach().cpu())
            trues.append(label.detach().cpu())
            losses.append(loss.detach().cpu())
        else:
            preds.append(prediction.detach())
            trues.append(label.detach())
            losses.append(loss.detach())
    test_time = timeit.default_timer() - start_time
    logger.info(f"Total test time: [{test_time / 60:2.4} mins]")
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


def LSTM_trainer(
        train_dataloader, val_dataloader, test_dataloader, vectors, in_dim: int = 100,
        hid_dim: int = 50, epoch=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        model_name='Glove'):
    # train_dataloader, test_dataloader = dataloaders
    model = BiLSTM_Emb_Classifier(
        vocab_size=vectors.shape[0], in_dim=in_dim, hid_dim=hid_dim, out_dim=cfg["data"]["num_classes"])
    # in_dim=in_dim, out_dim=cfg["data"]["num_classes"], hid_dim=hid_dim)
    logger.info(model)
    count_parameters(model)

    logger.info('Initialize the pretrained embedding')
    model.embedding.weight.data.copy_(vectors)

    if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # model_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    # model_name = model_name + '_epoch' + str(epoch)
    saved_model = load_model_state(model, model_name=model_name + '_epoch' + str(epoch), optimizer=optimizer)

    train_epochs_output_dict = None
    if not saved_model:
        logger.info(f'Model name: {model_name}')
        epoch_losses, train_epochs_output_dict = train_lstm_classifier(
            model, train_dataloader, loss_func=loss_func, optimizer=optimizer,
            epoch=epoch, eval_dataloader=val_dataloader,
            test_dataloader=test_dataloader, model_name=model_name)

    if saved_model:
        start_time = timeit.default_timer()
        losses, test_output = eval_lstm_classifier(
            saved_model, loss_func=loss_func, dataloader=test_dataloader)
        test_time = timeit.default_timer() - start_time
    else:
        start_time = timeit.default_timer()
        losses, test_output = eval_lstm_classifier(
            model, loss_func=loss_func, dataloader=test_dataloader)
        test_time = timeit.default_timer() - start_time

    test_count = test_dataloader.dataset.__len__()
    logger.info(f"Total inference time for [{test_count}] examples: [{test_time:2.4} sec]"
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
    LSTM_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
