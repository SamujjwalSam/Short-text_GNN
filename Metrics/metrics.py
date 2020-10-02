# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Metrics using PyTorch_Lightning
__description__ : Metrics for model performance calculation
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

import torch
import numpy as np
import pandas as pd
from pytorch_lightning.metrics.functional import f1_score as f1, precision, recall, accuracy as accuracy_pl
from pytorch_lightning.metrics.sklearns import F1, Accuracy, Precision, Recall


from sklearn.metrics import accuracy_score, recall_score, precision_score,\
    f1_score, precision_recall_fscore_support, classification_report


def calculate_performance_sk(true: np.ndarray, pred: np.ndarray) -> dict:
    """

    Parameters
    ----------
    true: Multi-hot
    pred: Multi-hot

    Returns
    -------

    """
    scores = {"accuracy": {}}
    scores["accuracy"]["unnormalize"] = accuracy_score(true, pred)
    scores["accuracy"]["normalize"] = accuracy_score(true, pred, normalize=True)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision_score(true, pred,
                                                     average=None).tolist()
    scores["precision"]["weighted"] = precision_score(true, pred,
                                                      average='weighted')
    scores["precision"]["micro"] = precision_score(true, pred, average='micro')
    scores["precision"]["macro"] = precision_score(true, pred, average='macro')
    scores["precision"]["samples"] = precision_score(true, pred,
                                                     average='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall_score(true, pred,
                                               average=None).tolist()
    scores["recall"]["weighted"] = recall_score(true, pred, average='weighted')
    scores["recall"]["micro"] = recall_score(true, pred, average='micro')
    scores["recall"]["macro"] = recall_score(true, pred, average='macro')
    scores["recall"]["samples"] = recall_score(true, pred, average='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_score(true, pred, average=None).tolist()
    scores["f1"]["weighted"] = f1_score(true, pred, average='weighted')
    scores["f1"]["micro"] = f1_score(true, pred, average='micro')
    scores["f1"]["macro"] = f1_score(true, pred, average='macro')
    scores["f1"]["samples"] = f1_score(true, pred, average='samples')

    return scores


def flatten_results(results: dict):
    """ Flattens the nested result dict and save as csv.

    :param results:
    :return:
    """
    ## Replace classes list to dict:
    for i, result in enumerate(results):
        for approach, vals in result.items():
            if approach != 'params':
                for metric1, averaging in vals.items():
                    for avg, score in averaging.items():
                        if avg == 'classes':
                            classes_dict = {}
                            for cls, val in enumerate(score):
                                cls = str(cls)
                                classes_dict[cls] = val
                            results[i][approach][metric1][avg] = classes_dict

    result_df = pd.json_normalize(results, sep='_')

    ## Round values and save:
    # result_df.round(decimals=4).to_csv('results.csv')

    return result_df


def calculate_performance_pl(true: torch.tensor, pred: torch.tensor) -> dict:
    """

    Parameters
    ----------
    true: Multi-hot
    pred: Multi-hot

    Returns
    -------

    """
    scores = {}
    scores["accuracy"] = accuracy_pl(true, pred)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision(true, pred, reduction='none')
    # scores["precision"]["micro"] = precision(true, pred, class_reduction='micro')
    # scores["precision"]["macro"] = precision(true, pred, reduction='macro')
    # scores["precision"]["weighted"] = precision(true, pred, reduction='weighted')
    # scores["precision"]["samples"] = precision(true, pred, reduction='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall(true, pred, reduction='none')
    # scores["recall"]["micro"] = recall(true, pred, reduction='micro')
    # scores["recall"]["macro"] = recall(true, pred, reduction='macro')
    # scores["recall"]["weighted"] = recall(true, pred, reduction='weighted')
    # scores["recall"]["samples"] = recall(true, pred, reduction='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1(true, pred, reduction='none')
    # scores["f1"]["micro"] = f1_score(true, pred, reduction='micro')
    # scores["f1"]["macro"] = f1_score(pred, true, reduction='macro')
    # scores["f1"]["weighted"] = f1_score(true, pred, reduction='weighted')
    # scores["f1"]["samples"] = f1_score(true, pred, reduction='samples')

    return scores


def calculate_performance_pl_sk(true: torch.tensor, pred: torch.tensor) -> dict:
    """

    Parameters
    ----------
    true: Multi-hot
    pred: Multi-hot

    Returns
    -------

    """
    scores = {}

    accuracy = Accuracy()
    scores["accuracy"] = accuracy(true, pred)

    # Precision
    precision_classes = Precision(average=None)
    precision_weighted = Precision(average='weighted')
    precision_micro = Precision(average='micro')
    precision_macro = Precision(average='macro')
    precision_samples = Precision(average='samples')

    scores["precision"] = {}
    scores["precision"]["classes"] = precision_classes(true, pred)
    scores["precision"]["weighted"] = precision_weighted(true, pred,)
    scores["precision"]["micro"] = precision_micro(true, pred)
    scores["precision"]["macro"] = precision_macro(true, pred)
    scores["precision"]["samples"] = precision_samples(true, pred)

    # Recall
    recall_classes = Recall(average=None)
    recall_weighted = Recall(average='weighted')
    recall_micro = Recall(average='micro')
    recall_macro = Recall(average='macro')
    recall_samples = Recall(average='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall_classes(true, pred)
    scores["recall"]["weighted"] = recall_weighted(true, pred)
    scores["recall"]["micro"] = recall_micro(true, pred)
    scores["recall"]["macro"] = recall_macro(true, pred)
    scores["recall"]["samples"] = recall_samples(true, pred)

    # F1
    f1_classes = F1(average=None)
    f1_weighted = F1(average='weighted')
    f1_micro = F1(average='micro')
    f1_macro = F1(average='macro')
    f1_samples = F1(average='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_classes(true, pred)
    scores["f1"]["weighted"] = f1_weighted(true, pred)
    scores["f1"]["micro"] = f1_micro(true, pred)
    scores["f1"]["macro"] = f1_macro(true, pred)
    scores["f1"]["samples"] = f1_samples(true, pred)

    return scores


def main():
    """ Main module to start code

    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    true = torch.tensor([[0, 1], [1, 0]])
    pred = torch.tensor([[0, 1], [0, 0]])
    pl_dict = calculate_performance_pl(true, pred)
    print(pl_dict)
    pl_sk_dict = calculate_performance_pl_sk(true, pred)
    print(pl_sk_dict)
    sk_dict = calculate_performance_sk(true.numpy(), pred.numpy())
    print(sk_dict)


if __name__ == "__main__":
    main()
