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

from Disaster_Models.BiLSTM_Classifier import BiLSTM_Classifier
from Disaster_Models.BOW_mean_Classifier import BOW_mean_Classifier
from Disaster_Models.CNN_Classifier import CNN_Classifier
from Disaster_Models.DenseCNN_Classifier import DenseCNN_Classifier
from Disaster_Models.FastText_Classifier import FastText_Classifier
from Disaster_Models.XML_CNN_Classifier import XMLCNN_Classifier
from Layers.bilstm_classifiers import BiLSTM_Emb_Classifier
from DPCNN.DPCNN import DPCNN
from Data_Handlers.create_datasets import prepare_single_dataset
from File_Handlers.json_handler import load_json
from Metrics.metrics import calculate_performance_sk as calculate_performance,\
    calculate_performance_bin_sk
from Utils.utils import logit2label, count_parameters
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user,\
    pretrain_dir, cuda_device

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


def train_disaster_classifier(
        model, dataloader: utils.data.dataloader.DataLoader,
        loss_func: nn.modules.loss.BCEWithLogitsLoss, optimizer,
        epoch: int = cfg['training']['num_epoch'],
        eval_dataloader: utils.data.dataloader.DataLoader = None,
        test_dataloader: utils.data.dataloader.DataLoader = None,
        class_names=cfg['data']['class_names'], model_name='Glove',
        scheduler=None, clf_type='disaster', pad_token_id=1, embeds=None,
        fix_len=None, run_eval=True):
    logger.info(f"Training {model_name} for {epoch} epoch: ")
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
            text, text_lengths = batch.text
            label = stack([batch.__getattribute__(cls) for cls in class_names]).T
            if clf_type == 'BiLSTMEmb':
                prediction = model(text, text_lengths.long().cpu())
            elif clf_type == 'DPCNN':
                text = embeds(text)
                prediction = model(text)
            else:
                pad_mask = ~(text == pad_token_id)
                ## Need to do .float() in separate step, else code fails:
                pad_mask = pad_mask.float()
                prediction = model(text, pad_mask)
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
        test_output = eval_all(model, loss_func=loss_func,
                               classifier_type=classifier_type, embeds=embeds,
                               fix_len=fix_len)
        if test_dataloader is not None:
            val_losses, val_output = eval_disaster_classifier(
                model, loss_func=loss_func, dataloader=test_dataloader,
                classifier_type=classifier_type, embeds=embeds, print_logits=True)
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
        f"{classifier_type} Epoch {max_result['epoch']}, MAX Score {max_result['score']:1.4} MAX Model {model_name}")
    return model, train_epoch_losses, train_epoch_dict


def eval_disaster_classifier(
        model, loss_func, dataloader: utils.data.dataloader.DataLoader,
        class_names=cfg['data']['class_names'], pad_token_id=1,
        classifier_type='disaster', embeds=None):
    # if use_saved:
    #     model = load_model_state(model, epoch)
    model.eval()
    preds = []
    trues = []
    losses = []
    # start_time = timeit.default_timer()
    for iter, batch in enumerate(dataloader):
        text, text_lengths = batch.text
        ## aggregate labels:
        label = stack([batch.__getattribute__(cls) for cls in class_names]).T
        if classifier_type == 'BiLSTMEmb':
            prediction = model(text, text_lengths.long().cpu())
        elif classifier_type == 'DPCNN':
            text = embeds(text)
            prediction = model(text)
        else:
            pad_mask = ~(text == pad_token_id)
            ## Need to do .float() in separate step, else code fails:
            pad_mask = pad_mask.float()
            prediction = model(text, pad_mask)
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

    if print_logits:
        logger.debug(preds)

    ## Converting raw scores to probabilities using Sigmoid:
    preds = sigmoid(preds)

    if print_logits:
        logger.debug(preds)

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


def eval_all(model, loss_func, class_names=cfg['data']['class_names'],
             test_files=cfg['data']['all_test_files'], classifier_type='disaster', embeds=None, fix_len=None):
    global all_test_dataloaders

    if all_test_dataloaders is None:
        all_test_dataloaders = {}
        for tfile in test_files:
            if classifier_type == 'DPCNN':
                dataset, vocab, dataloader = prepare_single_dataset(
                    data_dir=pretrain_dir, dataname=tfile + ".csv",
                    fix_len=fix_len)
            else:
                dataset, vocab, dataloader = prepare_single_dataset(
                    data_dir=pretrain_dir, dataname=tfile + ".csv")
            all_test_dataloaders[tfile] = dataloader

    all_test_output = {}
    for tfile in test_files:
        test_losses, test_output = eval_disaster_classifier(
            model, loss_func=loss_func,
            dataloader=all_test_dataloaders[tfile], n_classes=n_classes,
            classifier_type=classifier_type, embeds=embeds)
        all_test_output[tfile] = test_output['result']['f1_weighted']

    return all_test_output


def set_param_hypers(model):
    decay_parameters = []
    fine_tune_decay_parameters = []
    no_decay_parameters = []
    fine_tune_no_decay_parameters = []

    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            fine_tune_no_decay_parameters.append(param)
        else:
            if 'bias' in name.lower():
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

    # parameters = decay_parameters+no_decay_parameters +\
    #          fine_tune_decay_parameters+fine_tune_no_decay_parameters

    return decay_parameters, fine_tune_decay_parameters,\
           no_decay_parameters, fine_tune_no_decay_parameters


def get_optimizer(model, clf_type, lr=cfg["model"]["optimizer"]["lr"],
                  model_config=None, epoch=cfg["training"]["num_epoch"]):
    if clf_type == 'DPCNN':
        optimizer = optim.SGD(
            model.parameters(), lr=lr,
            momentum=cfg["model"]["optimizer"]["momentum"],
            weight_decay=cfg["model"]["optimizer"]["weight_decay"])

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int((4 / 5) * epoch)], gamma=0.1)

    elif clf_type == 'BiLSTMEmb':
        optimizer = optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=cfg["model"]["optimizer"]["weight_decay"])

        scheduler = None
    else:
        def lambda_(epoch):
            return (1 / 10) ** epoch

        decay_parameters, fine_tune_decay_parameters, no_decay_parameters,\
        fine_tune_no_decay_parameters = set_param_hypers(model)

        if model_config is None:
            model_config = load_json(filename=clf_type + '_config',
                                     filepath='Disaster_Model_Configs')

        optimizer = optim.AdamW([{'params':       decay_parameters,
                                  'lr':           model_config["lr"],
                                  'weight_decay': model_config["wd"]},
                                 {'params': no_decay_parameters,
                                  'lr':     model_config["lr"], 'weight_decay': 0},
                                 {'params':       fine_tune_decay_parameters,
                                  'lr':           model_config["fine_tune_lr"],
                                  'weight_decay': model_config["wd"]},
                                 {'params':       fine_tune_no_decay_parameters,
                                  'lr':           model_config["fine_tune_lr"],
                                  'weight_decay': 0}],
                                lr=lr)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lambda_] * 4)

    return optimizer, scheduler


def disaster_trainer(
        train_dataloader, val_dataloader, test_dataloader, vectors, classifier,
        clf_type, in_dim=300, hid_dim=128, epoch=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        model_name=None, pretrain_dataloader=None, fix_len=None, pad_idx=1,
        pretrain_epoch=cfg['training']['num_epoch'], examcon_pretrain=True,
        init_vectors=True, multi_gpu=False, embeds=None, use_kd=True):
    model_name = clf_type + '_' + model_name
    model_config = load_json(filename=clf_type + '_config',
                             filepath='Disaster_Model_Configs')
    if clf_type == 'BiLSTMEmb':
        model = classifier(vocab_size=vectors.shape[0], in_dim=in_dim,
                           hid_dim=hid_dim, out_dim=cfg["data"]["num_classes"])
    elif clf_type == 'DPCNN':
        model = classifier(channel_size=model_config['channel_size'])
        embeds = nn.Embedding(vectors.shape[0], in_dim)
        fix_len = 40

    else:
        model = classifier(embeddings=vectors, pad_idx=pad_idx,
                           classes_num=len(cfg['data']['class_names']),
                           config=model_config, device=cuda_device)
    logger.info((clf_type, model))
    count_parameters(model)

    if init_vectors:
        if clf_type == 'BiLSTMEmb':
            model.bilstm_embedding.embedding.weight.data.copy_(vectors)
            logger.info(f'Initialized pretrained embedding of shape {model.bilstm_embedding.embedding.weight.shape}')
        elif clf_type == 'DPCNN':
            embeds.weight.data.copy_(vectors)
        else:
            model.embeddings.data.copy_(vectors)
            logger.info(f'Initialized pretrained embedding of shape {model.embeddings.data.shape}')

    if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
        model.to(cuda_device)
        if multi_gpu:
            model = nn.DataParallel(model)
        if embeds is not None:
            embeds.to(cuda_device)

    # model_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    # model_name = model_name + '_epoch' + str(epoch)
    # saved_model = load_model_state(model, model_name=model_name + '_epoch' + str(epoch), optimizer=optimizer)
    optimizer, scheduler = get_optimizer(model, clf_type, lr=lr,
                                         model_config=model_config, epoch=epoch)

    model_name = model_name + '_' + str(lr)

    if pretrain_dataloader is not None:
        model_name = model_name + '_cls_preepoch_' + str(pretrain_epoch)
        logger.info(f'Training classifier with all pretraining data with '
                    f'classification task for pretrain_epoch {pretrain_epoch}')
        model, epoch_losses, train_epochs_output_dict = train_disaster_classifier(
            model, pretrain_dataloader, loss_func=loss_func, optimizer=optimizer,
            epoch=pretrain_epoch, eval_dataloader=val_dataloader,
            test_dataloader=test_dataloader, model_name=model_name,
            scheduler=scheduler, classifier_type=classifier_type,
            embeds=embeds, fix_len=fix_len)

    model_name = model_name + '_epoch_' + str(epoch)

    # if not saved_model:
    logger.info(f'Model name: {model_name}')
    model, epoch_losses, train_epochs_output_dict = train_disaster_classifier(
        model, train_dataloader, loss_func=loss_func, optimizer=optimizer,
        epoch=epoch, eval_dataloader=val_dataloader, test_dataloader=test_dataloader, model_name=model_name,
        scheduler=scheduler, clf_type=clf_type, embeds=embeds, fix_len=fix_len)

    # test_count = test_dataloader.dataset.__len__()
    # logger.info(f"Total inference time for [{test_count}] examples: [{test_time:2.4} sec]"
    #             f"\nPer example: [{test_time / test_count} sec]")
    # logger.info(dumps(test_output['result'], indent=4))

    return train_epochs_output_dict


def run_all_disaster(train_dataloader, val_dataloader, test_dataloader, vectors,
                     in_dim=300, epoch=cfg['training']['num_epoch'],
                     lr=cfg["model"]["optimizer"]["lr"], model_name=None,
                     pretrain_dataloader=None, init_vectors=True):
    classifiers = {
        # 'DPCNN': DPCNN,
        # 'BiLSTM_Emb': BiLSTM_Emb_Classifier,
        'BiLSTM':     BiLSTM_Classifier,
        'BOW_mean':   BOW_mean_Classifier,
        'CNN':        CNN_Classifier,
        'DenseCNN':   DenseCNN_Classifier,
        # 'FastText':   FastText_Classifier,
        'XML_CNN':    XMLCNN_Classifier
    }

    for classifier_type, classifier in classifiers.items():
        _ = disaster_trainer(
            train_dataloader, val_dataloader, test_dataloader, vectors,
            classifier, classifier_type, in_dim=in_dim, epoch=epoch,
            loss_func=nn.BCEWithLogitsLoss(), lr=lr, model_name=model_name,
            pretrain_dataloader=pretrain_dataloader, pretrain_epoch=cfg['pretrain']['epoch'],
            init_vectors=init_vectors)


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    disaster_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
