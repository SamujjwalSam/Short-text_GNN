# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Code related to applying GNN from DGL library
__description__ : node and graph classification written in DGL library
__project__     : gcn
__classes__     : gcn
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
# import numpy as np
from os.path import join
from torch import utils, cuda, save, load, device
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.pretrain_losses import supervised_contrastive_loss
from Layers.gcn_classifiers import GCN
from Utils.utils import count_parameters, save_model_state, load_model_state, save_token2pretrained_embs
from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Logger.logger import logger

device = device('cuda' if cuda.is_available() else 'cpu')


def eval_gcn(model, A, X):
    model.eval()
    X = model(A, X).detach().cpu()
    return X


def train_gcn(model, A, X, optimizer, dataloader: utils.data.dataloader.DataLoader, epochs: int = 5, node_list=None,
              idx2str=None, save_epochs=cfg['pretrain']['save_epochs']):
    logger.info("Started GCN training...")
    train_epoch_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = timeit.default_timer()
        X_hat = model(A, X)
        loss = 0
        for iter, (x_idx, x_pos_idx, x_pos_wt, x_neg_idx, x_neg_wt) in enumerate(dataloader):
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
        logger.info(f'Epoch {epoch}, Time: {epoch_train_time / 60:6.3} mins, Loss: {epoch_loss}')
        train_epoch_losses.append(epoch_loss)

        if epoch in save_epochs:
            X_hat_eval = eval_gcn(model, A, X)
            # token2pretrained_embs = get_token2pretrained_embs(X_hat, node_list, idx2str)
            save_token2pretrained_embs(X_hat_eval, node_list, idx2str, epoch=epoch)
            logger.info(f'Saved pretrained embeddings for epoch {epoch}')

    return train_epoch_losses


def gcn_trainer(A, X, train_dataloader, in_dim: int = 300, hid_dim: int = 300,
                epoch=cfg['training']['num_epoch'], lr=cfg["pretrain"]["lr"], node_list=None, idx2str=None, model_type='GCN'):
    model = GCN(in_dim=in_dim, hid_dim=hid_dim, out_dim=hid_dim)
    # model = MLP_Model(in_dim=in_dim, hid_dim=hid_dim, out_dim=hid_dim)

    logger.info(model)
    count_parameters(model)

    if cfg['model']['use_cuda'][plat][user] and cuda.is_available():
        model.to(device)
        A = A.to(device)
        X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    saved_model = load_model_state(model, model_name=epoch, optimizer=optimizer, model_dir=model_dir,
                                   sub_dir='pretrained')
    epoch_losses = None
    if not saved_model:
        epoch_losses = train_gcn(model, A, X, optimizer, train_dataloader,
                                 epochs=epoch, node_list=node_list, idx2str=idx2str)

        save_model_state(model, 'GCN' + str(epoch), optimizer)

    if saved_model:
        X_hat = eval_gcn(saved_model, A, X)
    else:
        X_hat = eval_gcn(model, A, X)

    return epoch_losses, X_hat


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    pass


if __name__ == "__main__":
    main()
    save_path = join(dataset_dir, cfg['data']['name'] + '_model.pt')
    state = load(save_path)
    token_gcn.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
