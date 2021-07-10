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
from torch import nn, stack, utils, sigmoid, mean, cat, cuda, save
from os import environ
from json import dumps
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import DataLoader

from Data_Handlers.instance_handler_dgl import Instance_Dataset_DGL
from Data_Handlers.create_datasets import prepare_single_dataset
from Layers.glen_classifier import GLEN_Classifier
from Metrics.metrics import calculate_performance_sk as calculate_performance,\
    calculate_performance_bin_sk
from Utils.utils import logit2label, count_parameters
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user, dataset_dir, cuda_device

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])
# device_id, cuda_device = set_cuda_device()


def train_glen(model, G, X, dataloader: utils.data.dataloader.DataLoader,
               loss_func: nn.modules.loss.BCEWithLogitsLoss, optimizer,
               epoch: int = cfg['training']['num_epoch'],
               eval_dataloader: utils.data.dataloader.DataLoader = None,
               test_dataloader: utils.data.dataloader.DataLoader = None,
               n_classes=cfg['data']['num_classes'], model_name='Glove'):
    logger.info(f"Started training for {epoch} epoch: ")
    train_epoch_losses = []
    train_epoch_dict = OrderedDict()
    max_result = {'score': 0.0, 'epoch': 0}
    for epoch in range(1, epoch + 1):
        model.train()
        epoch_loss = 0
        preds = []
        trues = []
        start_time = timeit.default_timer()
        for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(dataloader):
            ## Store emb in a separate file as self_loop removes emb info:
            emb = graph_batch.ndata['emb']
            # graph_batch = dgl.add_self_loop(graph_batch)
            if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
                graph_batch = graph_batch.to(cuda_device)
                emb = emb.to(cuda_device)
            prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X)
            # if epoch == 30:
            #     from evaluations import get_freq_disjoint_token_vecs, plot
            #     glove_vecs = get_freq_disjoint_token_vecs(S_vocab, T_vocab, X)
            #     plot(glove_vecs)
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
        test_output = eval_all(model, G, X, loss_func=loss_func)
        # logger.info(test_output)
        if eval_dataloader is not None:
            val_losses, val_output = eval_glen(model, G, X, loss_func=loss_func,
                                               dataloader=eval_dataloader)
            logger.info(
                f"Epoch {epoch}, time: {train_time / 60:1.4} mins, Train loss: "
                f"{epoch_loss:1.6} Val W-F1 {val_output['result']['f1_weighted'].item():4.4}"
                f" TEST W-F1: {test_output[cfg['data']['test']]:4.4} Dataset "
                f"{cfg['data']['test']} \n{dumps(test_output, indent=4)} Model {model_name}")
        else:
            logger.info(
                f"Epoch {epoch}, time: {train_time / 60:1.4} mins, Train loss: "
                f"{epoch_loss} TEST W-F1: {test_output[cfg['data']['test']]:4.4}"
                f" Dataset {cfg['data']['test']} \n{dumps(test_output, indent=4)}"
                f" Model {model_name}")

        if max_result['score'] < test_output[cfg['data']['test']]:
            max_result['score'] = test_output[cfg['data']['test']]
            max_result['epoch'] = epoch
            # max_result['result'] = test_output['result']

        # test_output = eval_all(model, G, X, loss_func=loss_func)
        # val_losses, val_output = eval_glen(model, G, X, loss_func=loss_func, dataloader=eval_dataloader)
        # logger.info(f'val_output: \n{dumps(val_output["result"], indent=4)}')
        # test_losses, test_output = eval_glen(model, G, X, loss_func=loss_func, dataloader=test_dataloader)
        # logger.info(f'test_output: \n{dumps(test_output["result"], indent=4)}')
        # logger.info(f"Epoch {epoch}, Train loss {epoch_loss:1.4}, val loss "
        #             f"{val_losses:1.6}, test loss {test_losses:1.6}, Val W-F1 "
        #             f"{val_output['result']['f1_weighted'].item():1.4} GLEN Test W-F1"
        #             f" {test_output['result']['f1_weighted'].item():1.4}, Model {model_name} time: "
        #             f"{train_time / 60:3.2} mins")
        # if max_result['score'] < test_output['result']['f1_weighted'].item():
        #     max_result['score'] = test_output['result']['f1_weighted'].item()
        #     max_result['epoch'] = epoch
        #     max_result['result'] = test_output['result']
        # logger.info(f"Epoch {epoch}, Train loss {epoch_loss}, val loss "
        #             f"{val_losses}, Val Weighted F1 {val_output['result']['f1_weighted'].item()}")
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

    logger.info(
        f"GLEN: Epoch {max_result['epoch']}, MAX Model {model_name} MAX Score "
        f"{max_result['score']:1.4}"
        f" Result {dumps(max_result, indent=4)},")

    return model, max_result, train_epoch_dict


all_test_dataloaders = None


def eval_all(
        model: GLEN_Classifier, G, X, loss_func, n_classes=cfg['data']['num_classes'],
        test_files=cfg['data']['all_test_files']):

    global all_test_dataloaders

    if all_test_dataloaders is None:
        all_test_dataloaders = {}
        for tfile in test_files:
            # tfile = tfile + ".csv"
            dataset, vocab, dataloader = prepare_single_dataset(
                data_dir=dataset_dir, dataname=tfile+'.csv')

            val_instance_graphs = Instance_Dataset_DGL(
                dataset, vocab, tfile, class_names=cfg['data']['class_names'])

            dataloader = DataLoader(
                val_instance_graphs, batch_size=cfg['training']['eval_batch_size'],
                shuffle=True, collate_fn=val_instance_graphs.batch_graphs)
            all_test_dataloaders[tfile] = dataloader

    all_test_output = {}
    for tfile in test_files:
        test_losses, test_output = eval_glen(
            model, G, X, loss_func=loss_func,
            dataloader=all_test_dataloaders[tfile], n_classes=n_classes)
        all_test_output[tfile] = test_output['result']['accuracy_normalize']

    return all_test_output


def eval_glen(model: GLEN_Classifier, G, X, loss_func,
              dataloader: utils.data.dataloader.DataLoader,
              n_classes=cfg['data']['num_classes'], save_gcn_embs=False):
    model.eval()
    preds = []
    trues = []
    losses = []
    # start_time = timeit.default_timer()
    for iter, (graph_batch, local_ids, label, global_ids, node_counts) in enumerate(dataloader):
        ## Store emb in a separate file as self_loop removes emb info:
        emb = graph_batch.ndata['emb']
        # graph_batch = dgl.add_self_loop(graph_batch)
        if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
            graph_batch = graph_batch.to(cuda_device)
            emb = emb.to(cuda_device)
            # local_ids = local_ids.to(cuda_device)
            # node_counts = node_counts.to(cuda_device)
            # global_ids = global_ids.to(cuda_device)
            G = G.to(cuda_device)
            X = X.to(cuda_device)
        if save_gcn_embs:
            save(X, 'X_glove.pt')
        prediction = model(graph_batch, emb, local_ids, node_counts, global_ids, G, X, save_gcn_embs)
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


def GLEN_trainer(
        G, X, train_dataloader, val_dataloader, test_dataloader, in_dim: int = 100,
        hid_dim: int = 50, num_heads: int = 2, epoch=cfg['training']['num_epoch'],
        loss_func=nn.BCEWithLogitsLoss(), lr=cfg["model"]["optimizer"]["lr"],
        n_classes=cfg['data']['num_classes'], state=None, model_name='GLEN'):
    # train_dataloader, test_dataloader = dataloaders
    model = GLEN_Classifier(
        in_dim=in_dim, hid_dim=hid_dim, num_heads=num_heads,
        out_dim=hid_dim, num_classes=train_dataloader.dataset.num_labels,
        state=state)
    logger.info(model)
    count_parameters(model)
    if cfg['cuda']['use_cuda'][plat][user] and cuda.is_available():
        model.to(cuda_device)
        G = G.to(cuda_device)
        X = X.to(cuda_device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Load optimizer state and params:
    # if state is not None:
    #     optimizer.load_state_dict(state['optimizer'])

    model, max_result, train_epochs_output_dict = train_glen(
        model, G, X, train_dataloader, loss_func=loss_func, optimizer=optimizer,
        epoch=epoch, eval_dataloader=val_dataloader, test_dataloader=test_dataloader, model_name=model_name)

    # start_time = timeit.default_timer()
    # losses, test_output = eval_glen(model, G, X, loss_func=loss_func,
    #                                 dataloader=test_dataloader, save_gcn_embs=True)
    # test_time = timeit.default_timer() - start_time
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
    GLEN_trainer(in_dim=1, hid_dim=4, num_heads=2)


if __name__ == "__main__":
    main()
