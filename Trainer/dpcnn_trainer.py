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
from torch import nn, stack, utils, sigmoid, mean, cat, cuda
from json import dumps
from math import isnan
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader

# from Layers.pretrain_losses import supervised_contrastive_loss
from DPCNN.DPCNN import DPCNN
from Data_Handlers.create_datasets import prepare_single_dataset
from Metrics.metrics import calculate_performance_sk as calculate_performance,\
    calculate_performance_bin_sk
from Utils.utils import logit2label, count_parameters
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user,\
    pretrain_dir, cuda_device

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


def train_dpcnn_classifier(
        model: DPCNN, embeds,
        dataloader: utils.data.dataloader.DataLoader,
        loss_func: nn.modules.loss.BCEWithLogitsLoss, optimizer,
        epoch: int = cfg['training']['num_epoch'],
        eval_dataloader: utils.data.dataloader.DataLoader = None,
        test_dataloader: utils.data.dataloader.DataLoader = None,
        n_classes=cfg['data']['num_classes'], model_name='Glove',
        scheduler=None):
    logger.info(f"Started training for {epoch} epoch: ")
    max_result = {'score': 0.0, 'epoch': 0}
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    for epoch in range(1, epoch + 1):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        start_time = timeit.default_timer()
        for iter, batch in enumerate(dataloader):
            # logger.debug(batch)
            text, text_lengths = batch.text
            ## Get label based on number of classes:
            # if cfg['data']['class_names'] == 1:
            #     label = batch.__getattribute__('0').unsqueeze(1)
            # else:
            label = stack([batch.__getattribute__(cls) for cls in cfg['data']['class_names']]).T
            text = embeds(text)
            # logger.debug(text.shape)
            prediction = model(text).squeeze()
            if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
                prediction = prediction.to(cuda_device)
                label = label.to(cuda_device)
            if prediction.dim() == 1:
                prediction = prediction.unsqueeze(1)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_count = label.shape[0]
            # logger.info(f'Running (iteration) loss: {loss.item():4.6}')
            if isnan(loss.item()):
                logger.fatal(f'Loss is {loss.item()} at iter {iter}, epoch {epoch}')
                logger.info("Named Parameters:\n")
                for name, param in model.named_parameters():
                    if param.requires_grad is True:
                        logger.info((name, param.size(), param))
                raise ValueError(f'Loss == NaN for model {model_name} at iter {iter}, epoch {epoch}')
            epoch_loss += loss.item()
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
        # if epoch == 10:
        #     print('here')
        test_output = eval_all(model, embeds, loss_func=loss_func)
        if eval_dataloader is not None:
            val_losses, val_output = eval_dpcnn_classifier(
                model, embeds, loss_func=loss_func, dataloader=eval_dataloader)
            logger.info(
                f"Epoch {epoch}, time: {train_time / 60:1.4} mins, Train loss: "
                f"{epoch_loss:0.6} Val W-F1 {val_output['result']['f1_weighted'].item():4.4}"
                f" TEST W-F1: {test_output[cfg['data']['test']]} Model {model_name}"
                f" Dataset {cfg['data']['test']}\n{dumps(test_output, indent=4)}")
        else:
            logger.info(
                f"Epoch {epoch}, time: {train_time / 60:1.4} mins, Train loss: "
                f"{epoch_loss} TEST W-F1: {test_output[cfg['data']['test']]}"
                f" Model {model_name} Dataset {cfg['data']['test']}"
                f"\n{dumps(test_output, indent=4)}")

        if max_result['score'] < test_output[cfg['data']['test']]:
            max_result['score'] = test_output[cfg['data']['test']]
            max_result['epoch'] = epoch
            # max_result['result'] = test_output['result']

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

        if scheduler is not None:
            scheduler.step()

    logger.info(
        f"dpcnn Epoch {max_result['epoch']}, MAX Model {model_name} MAX Score "
        f"{max_result['score']:1.4}")
    return model, train_epoch_losses, train_epoch_dict


def eval_dpcnn_classifier(model: DPCNN, embeds, loss_func,
                          dataloader: utils.data.dataloader.DataLoader,
                          n_classes=cfg['data']['num_classes']):
    # if use_saved:
    #     model = load_model_state(model, epoch)
    model.eval()
    preds = []
    trues = []
    losses = []
    # start_time = timeit.default_timer()
    for iter, batch in enumerate(dataloader):
        text, text_lengths = batch.text
        ## Get label based on number of classes:
        if cfg['data']['class_names'] == 1:
            label = batch.__getattribute__('0').unsqueeze(1)
        else:
            label = stack([batch.__getattribute__(cls) for cls in cfg['data']['class_names']]).T
        text = embeds(text)
        prediction = model(text)
        # test_count = label.shape[0]
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)
        if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
            prediction = prediction.to(cuda_device)
            label = label.to(cuda_device)
        loss = loss_func(prediction, label)
        if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
            preds.append(prediction.detach().cpu())
            trues.append(label.detach().cpu())
            losses.append(loss.detach().cpu())
        else:
            preds.append(prediction.detach())
            trues.append(label.detach())
            losses.append(loss.detach())
    # test_time = timeit.default_timer() - start_time
    # logger.info(f"Total test time: [{test_time / 60:2.4} mins]")
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
        # 'preds':  preds,
        # 'trues':  trues,
        'result': result_dict
    }
    # logger.info(dumps(result_dict, indent=4))

    return losses, test_output


all_test_dataloaders = None


def eval_all(model: DPCNN, embeds, loss_func, n_classes=cfg['data']['num_classes'],
             test_files=cfg['data']['all_test_files']):
    global all_test_dataloaders

    if all_test_dataloaders is None:
        all_test_dataloaders = {}
        for tfile in test_files:
            dataset, vocab, dataloader = prepare_single_dataset(data_dir=pretrain_dir, dataname=tfile + ".csv",
                                                                fix_length=40)
            all_test_dataloaders[tfile] = dataloader

    all_test_output = {}
    for tfile in test_files:
        test_losses, test_output = eval_dpcnn_classifier(
            model, embeds, loss_func=loss_func,
            dataloader=all_test_dataloaders[tfile], n_classes=n_classes)
        all_test_output[tfile] = test_output['result']['f1_weighted']

    return all_test_output


def DPCNN_trainer(
        train_dataloader, val_dataloader, test_dataloader, vectors=None, in_dim=300,
        channel_size: int = 250, epoch=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        model_name='DPCNN', pretrain_dataloader=None,
        pretrain_epoch=cfg['pretrain']['epoch'], init_vectors=True):
    # train_dataloader, test_dataloader = dataloaders
    model = DPCNN(channel_size=channel_size)
    embeds = nn.Embedding(vectors.shape[0], in_dim)
    logger.info(model)
    count_parameters(model)

    if init_vectors:
        embeds.weight.data.copy_(vectors)
        logger.info(f'Initialized pretrained embedding of shape {embeds.weight.shape}')

    if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
        model.to(cuda_device)
        embeds.to(cuda_device)

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=cfg["model"]["optimizer"]["momentum"],
        weight_decay=cfg["model"]["optimizer"]["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int((4/5)*epoch)], gamma=0.1)


    # model_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    # model_name = model_name + '_epoch' + str(epoch)
    # saved_model = load_model_state(model, model_name=model_name + '_epoch' + str(epoch), optimizer=optimizer)

    if pretrain_dataloader is not None:
        model_name = model_name + '_cls_preepoch_' + str(pretrain_epoch)
        logger.info(f'Training classifier with all pretraining data with '
                    f'classification task for pretrain_epoch {pretrain_epoch}')
        model, epoch_losses, train_epochs_output_dict = train_dpcnn_classifier(
            model, pretrain_dataloader, loss_func=loss_func, optimizer=optimizer,
            epoch=pretrain_epoch, eval_dataloader=val_dataloader,
            test_dataloader=test_dataloader, model_name=model_name)

    model_name = model_name + '_epoch_' + str(epoch)

    train_epochs_output_dict = None
    # if not saved_model:
    logger.info(f'Model name: {model_name}')
    model, epoch_losses, train_epochs_output_dict = train_dpcnn_classifier(
        model, embeds, train_dataloader, loss_func=loss_func, optimizer=optimizer,
        epoch=epoch, eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader, model_name=model_name, scheduler=scheduler)

    # if saved_model:
    #     start_time = timeit.default_timer()
    #     losses, test_output = eval_dpcnn_classifier(
    #         saved_model, loss_func=loss_func, dataloader=test_dataloader)
    #     test_time = timeit.default_timer() - start_time
    # else:
    #     start_time = timeit.default_timer()
    #     losses, test_output = eval_dpcnn_classifier(
    #         model, loss_func=loss_func, dataloader=test_dataloader)
    #     test_time = timeit.default_timer() - start_time

    # test_count = test_dataloader.dataset.__len__()
    # logger.info(f"Total inference time for [{test_count}] examples: [{test_time:2.4} sec]"
    #             f"\nPer example: [{test_time / test_count} sec]")
    # logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    dpcnn_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
