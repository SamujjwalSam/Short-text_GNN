# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Losses for contrastive pretraining.
__description__ :
__project__     : WSCP
__classes__     :
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "10/01/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import torch
import torch.nn.functional as F


def contrastive_loss(x, x_pos, x_neg, delta=0.1):
    """ Calculate token level contrastive loss using dot product.

    :param x: Representation of node
    :param x_pos: Representation of neighbors from G+
    :param x_neg: Representation of neighbors from G-
    :param delta: (hyperparam) Minimum margin between positive and negative nodes.
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(1)
    if len(x_pos.shape) == 2:
        x_pos = x_pos.unsqueeze(1)
    if len(x_neg.shape) == 2:
        x_neg = x_neg.unsqueeze(1)
        ## TODO: Take weighted average
    dot_pos = torch.sum(torch.tensordot(x, x_pos, dims=([1, 2], [1, 2]))) / x_pos.shape[0]
    dot_neg = torch.sum(torch.tensordot(x, x_neg, dims=([1, 2], [1, 2]))) / x_neg.shape[0]
    ## TODO: Take log: log(pos) - log(neg)
    return torch.relu(dot_pos - dot_neg + delta)


def supervised_contrastive_loss(x, x_pos, x_neg, x_pos_wt=None, x_neg_wt=None, tau=5):
    """ Calculate token level contrastive loss using dot product.

    :param tau: Temperature for softmax
    :param x_neg_wt:
    :param x_pos_wt:
    :param x: Representation of node from G
    :param x_pos: Representation of neighbors from G+
    :param x_neg: Representation of all neighbors G+ U G-
    """
    if len(x.shape) == 2:
        x = x.unsqueeze(1)
    if len(x_pos.shape) == 2:
        x_pos = x_pos.unsqueeze(1)
    if len(x_neg.shape) == 2:
        x_neg = x_neg.unsqueeze(1)

    # x_all = torch.cat((x_pos, x_neg), 0)

    x_pos_dot = - torch.tensordot(x, x_pos, dims=([1, 2], [1, 2])) / tau
    x_neg_dot = - torch.tensordot(x, x_neg, dims=([1, 2], [1, 2])) / tau
    # logger.debug(f'POS neighbors: {x_pos.shape}, {x_pos_dot}')
    # logger.debug(f'NEG neighbors: {x_neg.shape}, {x_neg_dot}')

    # if x_pos_wt is not None or x_neg_wt is not None:
    #     x_all_wt = torch.tensor(x_pos_wt + x_neg_wt).to('cuda:0')
    #     x_dot = x_all_wt * x_dot

    result = F.log_softmax(torch.cat((x_pos_dot.T, x_neg_dot.T), 0), dim=0)[:x_pos.shape[0]]
    # nume = torch.tensordot(x, x_pos, dims=([1, 2], [1, 2]))
    # nume = torch.exp(nume)
    #
    # deno = torch.tensordot(x, x_all, dims=([1, 2], [1, 2]))
    # deno = torch.exp(deno)
    # deno = torch.sum(deno)
    #
    # result = torch.log(nume / deno)
    # logger.debug(result)
    result = result / x_pos.shape[0]
    result = -torch.sum(result)

    return result


def contrastive_distance(x, x_pos, x_neg, delta=0.6):
    """ Calculate token level contrastive loss using pairwise_distance.

    :param x: Representation of node
    :param x_pos: Representation of neighbors from G+
    :param x_neg: Representation of neighbors from G-
    :param delta: (hyperparam) Minimum margin between positive and negative nodes.
    """
    ## TODO: Take weighted average
    dot_pos = torch.sum(F.pairwise_distance(x, x_pos)) / x_pos.shape[0]
    dot_neg = torch.sum(F.pairwise_distance(x, x_neg)) / x_neg.shape[0]
    ## TODO: Take log: log(pos) - log(neg)
    return torch.relu(dot_pos - dot_neg + delta)


if __name__ == "__main__":
    epochs = 5
    d = 2
    w = 3
    n = d + w
    emb_dim = 2

    A = torch.randn(n, n)
    X = torch.randn(n, emb_dim)
    target = torch.randint(0, 2, (n, emb_dim)).float()

    mat_test = Pretrain_MLP(in_dim=2, hid_dim=2, out_dim=1)

    X_hat = mat_test(A, X)
