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
from os.path import join, exists
from torch import nn, stack, utils, sigmoid, mean, cat, cuda, save, load
from json import dumps
from math import isnan
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.pretrain_losses import supervised_contrastive_loss
from Data_Handlers.create_datasets import prepare_single_dataset
from Layers.bilstm_classifiers import BiLSTM_Emb_Classifier, BiLSTM_Emb_repr
from Metrics.metrics import calculate_performance_sk as calculate_performance
from Utils.utils import logit2label, count_parameters, save_model_state, load_model_state
from File_Handlers.read_file import get_latest_filename, is_empty_dir
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, cuda_device

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


def train_lstm_classifier(
        model: BiLSTM_Emb_Classifier, dataloader: utils.data.dataloader.DataLoader,
        loss_func: nn.modules.loss.BCEWithLogitsLoss, optimizer,
        epoch: int = cfg['training']['num_epoch'],
        eval_dataloader: utils.data.dataloader.DataLoader = None,
        test_dataloader: utils.data.dataloader.DataLoader = None,
        class_names=cfg['data']['class_names'], model_name='Glove',
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
            label = stack([batch.__getattribute__(cls) for cls in class_names]).T
            prediction = model(text, text_lengths.long().cpu()).squeeze()
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
        test_output = eval_all(model, loss_func=loss_func)
        if eval_dataloader is not None:
            val_losses, val_output = eval_lstm_classifier(
                model, loss_func=loss_func, dataloader=eval_dataloader)
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
        f"LSTM Epoch {max_result['epoch']}, MAX Model {model_name} MAX Score "
        f"{max_result['score']:1.4} ")
    return model, train_epoch_losses, train_epoch_dict


def train_lstm_examplecon(model, optimizer, neighbors_dataset,
                          example_dataloader: utils.data.dataloader.DataLoader,
                          epochs: int = 5, model_name='examplecon_lstm',
                          save_epochs=cfg['pretrain']['save_epochs']):
    logger.info(f"Started [{model_name}] training for {epochs} epochs: ")
    train_epoch_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_start_time = timeit.default_timer()
        X_hat = []
        for iter, batch in enumerate(example_dataloader):
            text, text_lengths = batch.text
            X_hat.append(model(text, text_lengths.long().cpu()))
        X_hat = cat(X_hat)
        loss = 0
        for iter, (x_idx, x_pos_idx, x_neg_idx) in enumerate(neighbors_dataset):
            x = X_hat[x_idx]
            x_pos = X_hat[x_pos_idx]
            x_neg = X_hat[x_neg_idx]
            if x.dim() == 1:
                x = x.unsqueeze(1).T
            if iter == 0:
                loss = supervised_contrastive_loss(x, x_pos, x_neg)
            else:
                loss += supervised_contrastive_loss(x, x_pos, x_neg)
            epoch_loss += loss.detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_time = timeit.default_timer() - epoch_start_time
        epoch_loss = loss.detach().item() / (iter + 1)
        logger.info(f'Epoch {epoch}, Time: {epoch_train_time / 60:6.3} mins, '
                    f'Loss: {epoch_loss} Model {model_name}')
        train_epoch_losses.append(epoch_loss)

        if epoch in save_epochs:
            save_name = model_name + str(epoch) + '_ecl'

            emb_save_name = save_name + '_embs'
            save(model.embedding.weight.data, join(dataset_dir, 'saved_models', emb_save_name))
            logger.info(f'Embeddings for epoch {epoch} saved at {emb_save_name}')

            # model_save_name = save_name + '_model'
            # save_model_state(model, model_save_name, optimizer,
            #                  model_dir=join(dataset_dir, 'saved_models'))
            logger.info(f'Model {emb_save_name} for epoch {epoch} saved at {join(dataset_dir, "saved_models", emb_save_name)}')

    return model


def eval_lstm_classifier(model: BiLSTM_Emb_Classifier, loss_func,
                         dataloader: utils.data.dataloader.DataLoader,
                         class_names=cfg['data']['class_names']):
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
        label = stack([batch.__getattribute__(cls) for cls in class_names]).T
        prediction = model(text, text_lengths.long().cpu())
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
    result_dict = calculate_performance(trues, preds)
    test_output = {
        # 'preds':  preds,
        # 'trues':  trues,
        'result': result_dict
    }
    # logger.info(dumps(result_dict, indent=4))

    return losses, test_output


all_test_dataloaders = None


def eval_all(model: BiLSTM_Emb_Classifier, loss_func,
             class_names=cfg['data']['class_names'], test_files=cfg['data']['all_test_files']):
    global all_test_dataloaders

    if all_test_dataloaders is None:
        all_test_dataloaders = {}
        for tfile in test_files:
            dataset, vocab, dataloader = prepare_single_dataset(data_dir=pretrain_dir, dataname=tfile + ".csv")
            all_test_dataloaders[tfile] = dataloader

    all_test_output = {}
    for tfile in test_files:
        test_losses, test_output = eval_lstm_classifier(
            model, loss_func=loss_func, dataloader=all_test_dataloaders[tfile], class_names=class_names)
        all_test_output[tfile] = test_output['result']['f1_weighted']

    return all_test_output


def get_final_logits(model, test_dataloader, loss_func=nn.BCEWithLogitsLoss()):
    _, logits = eval_lstm_classifier(
        model, loss_func=loss_func, dataloader=test_dataloader)

    ## Convert single class logit to multi-class proba:
    idxs = []
    logits0 = []
    logits1 = []
    for idx, (t, p) in logits:
        if t == 1:
            logits0.append(1 - p)
            logits1.append(p)
        else:
            logits0.append(p)
            logits1.append(1 - p)

    return idxs, logits0, logits1


def LSTM_trainer(
        train_dataloader, val_dataloader, test_dataloader, vectors, in_dim=300,
        hid_dim: int = 300, epoch=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        model_name='Glove', pretrain_dataloader=None,
        pretrain_epoch=cfg['pretrain']['epoch'], ecl_pretrain=True, init_vectors=True):
    # train_dataloader, test_dataloader = dataloaders
    model = BiLSTM_Emb_Classifier(vocab_size=vectors.shape[0], in_dim=in_dim,
                                  hid_dim=hid_dim, out_dim=cfg["data"]["num_classes"])
    logger.info(model)
    count_parameters(model)

    if init_vectors:
        model.bilstm_embedding.embedding.weight.data.copy_(vectors)
        logger.info(f'Initialized pretrained embedding of shape {model.bilstm_embedding.embedding.weight.shape}')

    if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
        model.to(cuda_device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=cfg["model"]["optimizer"]["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int((4 / 5) * epoch)], gamma=0.1)

    # model_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    # model_name = model_name + '_epoch' + str(epoch)
    # saved_model = load_model_state(model, model_name=model_name + '_epoch' + str(epoch), optimizer=optimizer)

    if pretrain_dataloader is not None:
        if ecl_pretrain:
            # ecl_model_name = model_name + '_ecl.pt'
            ecl_model_dir = join(dataset_dir, 'saved_models')
            if model_name.startswith('Glove'):
                filename_start = 'Glove_ecl_*'
            elif model_name.startswith('GCPD'):
                filename_start = 'GCPD_ecl_*'
            if not is_empty_dir(ecl_model_dir):
                latest_ecl_model_path = get_latest_filename(ecl_model_dir, filename_start=filename_start)
                logger.info(f'Loading ECL embeddings from [{latest_ecl_model_path}]')
                ecl_emb_data = load(latest_ecl_model_path)
            else:
                ecl_pretrain_epoch = cfg['pretrain']['epoch']
                ecl_model = BiLSTM_Emb_repr(vocab_size=vectors.shape[0], in_dim=in_dim,
                                            out_dim=hid_dim)
                if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
                    ecl_model.to(cuda_device)
                model_name = model_name + '_ecl_epoch_' + str(ecl_pretrain_epoch)
                # loss_func = nn.BCEWithLogitsLoss() ecl_pretrain
                logger.info(f'Training classifier with all pretraining data'
                            f' with contrastive task for pretrain_epoch '
                            f'{ecl_pretrain_epoch}')
                ecl_model = train_lstm_examplecon(
                    ecl_model, optimizer, pretrain_dataloader[0], pretrain_dataloader[1],
                    epochs=pretrain_epoch, model_name=model_name)
                ## Load pretrained embeddings to current model:
                ecl_emb_data = ecl_model.embedding.weight.data
                ecl_model_path = join(ecl_model_dir, model_name)
                save(ecl_emb_data, ecl_model_path)

            model.bilstm_embedding.embedding.weight.data.copy_(ecl_emb_data)
            # model.bilstm_embedding.load_state_dict(ecl_model.state_dict())
        else:
            model_name = model_name + '_cls_preepoch_' + str(pretrain_epoch)
            logger.info(f'Training classifier with all pretraining data with '
                        f'classification task for pretrain_epoch {pretrain_epoch}')
            model, epoch_losses, train_epochs_output_dict = train_lstm_classifier(
                model, pretrain_dataloader, loss_func=loss_func, optimizer=optimizer,
                epoch=pretrain_epoch, eval_dataloader=val_dataloader,
                test_dataloader=test_dataloader, model_name=model_name)

    model_name = model_name + '_epoch_' + str(epoch)

    train_epochs_output_dict = None
    # if not saved_model:
    logger.info(f'Model name: {model_name}')
    model, epoch_losses, train_epochs_output_dict = train_lstm_classifier(
        model, train_dataloader, loss_func=loss_func, optimizer=optimizer,
        epoch=epoch, eval_dataloader=val_dataloader,
        test_dataloader=test_dataloader, model_name=model_name, scheduler=scheduler)

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
    LSTM_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
