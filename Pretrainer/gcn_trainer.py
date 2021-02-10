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
from torch import utils, cuda, save, load, device  # , stack, norm, sum, Tensor
import torch.optim as optim
from torch.utils.data import DataLoader

from Layers.pretrain_models import supervised_contrastive_loss
from Layers.gcn_classifiers import GCN
from Utils.utils import count_parameters
from Logger.logger import logger
from config import configuration as cfg, platform as plat, username as user, dataset_dir

device = device('cuda' if cuda.is_available() else 'cpu')


def train_gcn(model, A, X, optimizer, dataloader: utils.data.dataloader.DataLoader, epochs: int = 5):
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
        logger.info(f'Epoch {epoch}, Time: {epoch_train_time / 60} mins, Loss: {epoch_loss}')
        train_epoch_losses.append(epoch_loss)

    return train_epoch_losses


def gcn_trainer(A, X, train_dataloader, in_feats: int = 300, hid_feats: int = 300,
                epochs=cfg['training']['num_epoch'], lr=cfg["pretrain"]["lr"]):
    model = GCN(in_dim=in_feats, hid_dim=hid_feats, out_dim=hid_feats)
    # model = Pretrain_MLP(in_dim=in_feats, hid_dim=hid_feats, out_dim=hid_feats)

    logger.info(model)
    count_parameters(model)

    model.to(device)
    A = A.to(device)
    X = X.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = train_gcn(
        model, A, X, optimizer, train_dataloader, epochs=epochs)

    # https://stackoverflow.com/a/49078976/2794244
    state = {
        'epoch':      epochs,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }
    save_path = join(join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name']),
                     cfg['data']['name'] + '_model' + str(
                         epochs) + '.pt')

    save(state, save_path)

    model.eval()
    X_hat = model(A, X)
    X_hat = X_hat.detach().cpu().numpy()

    return epoch_losses, state, save_path, X_hat


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
