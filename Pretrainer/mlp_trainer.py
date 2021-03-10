# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Code related to applying GNN from DGL library
__description__ : node and graph classification written in DGL library
__project__     : GCPD
__classes__     : GCPD
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "05/02/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import timeit
from os.path import join
import torch.optim as optim
from torch import utils, cuda, save, device, stack
from torch.utils.data import DataLoader

from Layers.pretrain_losses import supervised_contrastive_loss
from Layers.mlp_classifier import MLP_Model
from Utils.utils import count_parameters, save_token2pretrained_embs
from config import configuration as cfg, platform as plat, username as user, pretrain_dir, device
from Logger.logger import logger

## Enable multi GPU cuda environment:

if cuda.is_available():
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


def eval_mlp(model, X):
    model.eval()
    X = model(X).detach().cpu()
    # embs = []
    # start_time = timeit.default_timer()
    # for iter, x in enumerate(X):
    #     x = model(x)
    #     if x.dim() == 1:
    #         x = x.unsqueeze(1).T
    #     embs.append(x.detach().cpu())
    # test_time = timeit.default_timer() - start_time
    # logger.info(f"Total test time: [{test_time / 60:.4f} mins]")

    # return stack(embs).squeeze().numpy()
    return X


def train_mlp(model, X, optimizer, dataloader: utils.data.dataloader.DataLoader, epoch: int = 5, node_list=None, idx2str=None, save_epochs=cfg['pretrain']['save_epochs']):
    logger.info(f"Started MLP training for {epoch} epoch.")
    train_epoch_losses = []
    train_epoch_embs = []
    train_epoch_weights = []
    params, grads = None, None
    x_hat_old = None
    model.train()
    for epoch in range(epoch):
        epoch_loss = 0
        epoch_start_time = timeit.default_timer()
        loss = 0
        for iter, (x_idx, x_pos_idx, x_pos_wt, x_neg_idx, x_neg_wt) in enumerate(dataloader):
            x_all_idx = x_pos_idx + x_neg_idx + [x_idx]
            # x_all_wt = x_pos_wt + x_neg_wt
            # x_neg_idx.extend(x_pos_idx)
            # x_neg_idx.append(x_idx)
            # x = X[x_idx].unsqueeze(0)
            # x_pos = X[x_pos_idx]
            x_all = X[x_all_idx]
            # x_hat = model(x)
            # print_diff_norm(x, x_hat)
            # x_pos_hat = model(x_pos)
            # print_diff_norm(x_pos, x_pos_hat)
            x_all_hat = model(x_all)
            x_hat = x_all_hat[-1, :].unsqueeze(0)
            x_pos_hat = x_all_hat[:len(x_pos_idx), :]
            x_neg_hat = x_all_hat[len(x_pos_idx):len(x_pos_idx)+len(x_neg_idx), :]
            if x_hat.dim() == 1:
                x_hat = x_hat.unsqueeze(1).T
            # if iter == 0:
            # loss = supervised_contrastive_loss(x_hat, x_pos_hat, x_neg_hat, x_pos_wt, x_neg_wt)
            loss = supervised_contrastive_loss(x_hat, x_pos_hat, x_neg_hat, None, None)

            # print_similarities(x_hat, x_pos_hat, x_neg_hat)

            # x_hat_old = print_diff_norm(x_hat, x_hat_old)
            # else:
            #     loss += supervised_contrastive_loss(x, x_pos, x_neg)
            # iter_train_time = timeit.default_timer() - iter_start_time
            # logger.info(f"Training time per iter: [{iter_train_time}]")
            # logger.info(f"Iteration {iter}, loss: {loss.detach().item()}")
            optimizer.zero_grad()
            loss.backward()
            # logger.debug(f'LOSS: {loss}')
            # params, grads = print_norms(model, params, grads)
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_train_time = timeit.default_timer() - epoch_start_time
        example_loss = loss.detach().item() / (iter + 1)
        epoch_loss = loss.detach().item()
        logger.info(f'Epoch {epoch}, Time: {epoch_train_time / 60:.4f} mins, '
                    f'Epoch Loss: {epoch_loss:6.6f}, Loss/Example: {example_loss:2.8f}')
        train_epoch_losses.append(epoch_loss)

        if epoch in save_epochs:
            X_hat_eval = eval_mlp(model, X)
            # token2pretrained_embs = get_token2pretrained_embs(X_hat, node_list, idx2str)
            save_token2pretrained_embs(X_hat_eval, node_list, idx2str, epoch=epoch)
            logger.info(f'Saved pretrained embeddings for epoch {epoch}')

    return train_epoch_losses


def mlp_trainer(X, train_dataloader, in_dim: int = 300, hid_dim: int = 300,
                epoch=cfg['training']['num_epoch'], lr=cfg["pretrain"]["lr"],node_list=None, idx2str=None):
    model = MLP_Model(in_dim=in_dim, hid_dim=hid_dim, out_dim=in_dim, num_layer=2)

    logger.info(model)
    count_parameters(model)

    model.to(device)
    X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = train_mlp(
        model, X, optimizer, train_dataloader, epoch=epoch, node_list=node_list, idx2str=idx2str)

    # https://stackoverflow.com/a/49078976/2794244
    state = {
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }
    save_path = join(join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name']),
                     cfg['data']['name'] + '_model' + str(epoch) + '.pt')

    save(state, save_path)

    # model.eval()
    # X_hat = model(A, X)
    X_hat = eval_mlp(model, X)

    return epoch_losses, state, save_path, X_hat
