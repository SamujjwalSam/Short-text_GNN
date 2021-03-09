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
from json import dumps
# from pytorch_lightning.metrics.functional import f1_score as f1_pl, precision, recall, accuracy as accuracy_pl
from sklearn.metrics import accuracy_score, recall_score, precision_score,\
    f1_score, precision_recall_fscore_support, classification_report

from Logger.logger import logger


def calculate_performance_sk(true: (np.ndarray, torch.tensor), pred: (np.ndarray, torch.tensor), print_result=False) -> dict:
    """

    :param pred: Multi-hot
    :param true: Multi-hot
    :param print_result:

    """
    scores = {"accuracy": {}}
    scores["accuracy"]["unnormalize"] = accuracy_score(true, pred)
    scores["accuracy"]["normalize"] = accuracy_score(true, pred, normalize=True)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision_score(true, pred, average=None).tolist()
    scores["precision"]["weighted"] = precision_score(true, pred, average='weighted')
    scores["precision"]["macro"] = precision_score(true, pred, average='macro')
    scores["precision"]["micro"] = precision_score(true, pred, average='micro')
    scores["precision"]["samples"] = precision_score(true, pred, average='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall_score(true, pred, average=None).tolist()
    scores["recall"]["weighted"] = recall_score(true, pred, average='weighted')
    scores["recall"]["macro"] = recall_score(true, pred, average='macro')
    scores["recall"]["micro"] = recall_score(true, pred, average='micro')
    scores["recall"]["samples"] = recall_score(true, pred, average='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_score(true, pred, average=None).tolist()
    scores["f1"]["weighted"] = f1_score(true, pred, average='weighted')
    scores["f1"]["macro"] = f1_score(true, pred, average='macro')
    scores["f1"]["micro"] = f1_score(true, pred, average='micro')
    scores["f1"]["samples"] = f1_score(true, pred, average='samples')

    if print_result:
        logger.info(dumps(scores, indent=4))

    return scores


def calculate_performance_bin_sk(true: (np.ndarray, torch.tensor), pred: (np.ndarray, torch.tensor), print_result=False) -> dict:
    """

    :param pred: Multi-hot
    :param true: Multi-hot
    :param print_result:

    """
    scores = {"accuracy": {}}
    scores["accuracy"]["unnormalize"] = accuracy_score(true, pred)
    scores["accuracy"]["normalize"] = accuracy_score(true, pred, normalize=True)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision_score(true, pred, average=None).tolist()
    scores["precision"]["weighted"] = precision_score(true, pred, average='weighted')
    scores["precision"]["macro"] = precision_score(true, pred, average='macro')
    scores["precision"]["micro"] = precision_score(true, pred, average='micro')
    # scores["precision"]["samples"] = precision_score(true, pred, average='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall_score(true, pred, average=None).tolist()
    scores["recall"]["weighted"] = recall_score(true, pred, average='weighted')
    scores["recall"]["macro"] = recall_score(true, pred, average='macro')
    scores["recall"]["micro"] = recall_score(true, pred, average='micro')
    # scores["recall"]["samples"] = recall_score(true, pred, average='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_score(true, pred, average=None).tolist()
    scores["f1"]["weighted"] = f1_score(true, pred, average='weighted')
    scores["f1"]["macro"] = f1_score(true, pred, average='macro')
    scores["f1"]["micro"] = f1_score(true, pred, average='micro')
    # scores["f1"]["samples"] = f1_score(true, pred, average='samples')

    if print_result:
        logger.info(dumps(scores, indent=4))

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


def calculate_performance_pl(true: torch.tensor, pred: torch.tensor,
                             return_list=True, print_result=False) -> dict:
    """

    :param print_result:
    :param true: Multi-hot
    :param pred: Multi-hot
    :param return_list:

    """
    scores = {}
    scores["accuracy"] = accuracy_pl(true, pred)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision(true, pred, class_reduction='none')
    scores["precision"]["weighted"] = precision(true, pred, class_reduction='weighted')
    scores["precision"]["macro"] = precision(true, pred, class_reduction='macro')
    scores["precision"]["micro"] = precision(true, pred, class_reduction='micro')
    # scores["precision"]["samples"] = precision(true, pred, reduction='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall(true, pred, class_reduction='none')
    scores["recall"]["weighted"] = recall(true, pred, class_reduction='weighted')
    scores["recall"]["macro"] = recall(true, pred, class_reduction='macro')
    scores["recall"]["micro"] = recall(true, pred, class_reduction='micro')
    # scores["recall"]["samples"] = recall(true, pred, reduction='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_pl(true, pred, class_reduction='none')
    scores["f1"]["weighted"] = f1_pl(true, pred, class_reduction='weighted')
    scores["f1"]["macro"] = f1_pl(pred, true, class_reduction='macro')
    scores["f1"]["micro"] = f1_pl(true, pred, class_reduction='micro')
    # scores["f1"]["samples"] = f1_pl(true, pred, reduction='samples')

    if return_list:
        scores["accuracy"] = scores["accuracy"].tolist()

        scores["precision"]["classes"] = scores["precision"]["classes"].tolist()
        scores["precision"]["weighted"] = scores["precision"]["weighted"].tolist()
        scores["precision"]["macro"] = scores["precision"]["macro"].tolist()
        scores["precision"]["micro"] = scores["precision"]["micro"].tolist()

        scores["recall"]["classes"] = scores["recall"]["classes"].tolist()
        scores["recall"]["weighted"] = scores["recall"]["weighted"].tolist()
        scores["recall"]["macro"] = scores["recall"]["macro"].tolist()
        scores["recall"]["micro"] = scores["recall"]["micro"].tolist()

        scores["f1"]["classes"] = scores["f1"]["classes"].tolist()
        scores["f1"]["weighted"] = scores["f1"]["weighted"].tolist()
        scores["f1"]["macro"] = scores["f1"]["macro"].tolist()
        scores["f1"]["micro"] = scores["f1"]["micro"].tolist()

        if print_result:
            logger.info(dumps(scores, indent=4))
    else:
        if print_result:
            logger.info(scores)

    return scores


def precision_at_k(actuals, predictions, k=5, pos_label=1):
    """
    Function to evaluate the precision @ k for a given
    ground truth vector and a list of predictions (between 0 and 1).

    Args:
        actuals : np.array consisting of multi-hot encoding of label vector
        predictions : np.array consisting of predictive probabilities for every label.
        k : Value of k. Default: 5
        pos_label : Value to consider as positive. Default: 1

    Returns:
        precision @ k for a given ground truth - prediction pair.
    """
    assert len(actuals) == len(predictions),\
        f"P@k: Length mismatch: len(actuals) [{len(actuals)}] != [{len(predictions)}] len(predictions)"

    ## Converting to Numpy as it has supported funcions.
    if torch.is_tensor(actuals):
        logger.info(f"'actuals' is of [{type(actuals)}] type. Converting to Numpy.")
        actuals = actuals.numpy()
        logger.info(actuals)
    if torch.is_tensor(predictions):
        logger.info(f"'predictions' is of [{type(predictions)}] type. Converting to Numpy.")
        predictions = predictions.data.numpy()
        logger.info(predictions)

    n_pos_vals = (actuals == pos_label).sum()
    desc_order = np.argsort(predictions, -k)  # [::-1] reverses array
    matches = np.take(actuals, desc_order[:, :k])  # taking the top indices
    relevant_preds = (matches == pos_label).sum()

    return relevant_preds / min(n_pos_vals, k)


def precision_k_hot(actuals: torch.Tensor, predictions: torch.Tensor,
                    k: int = 1, pos_label: int = 1) -> float:
    """
    Calculates precision of actuals multi-hot vectors and predictions probabilities of shape: (batch_size,
    Number of samples, Number of categories).

    :param actuals: 3D torch.tensor consisting of multi-hot encoding of label vector of shape: (batch_size,
    Number of samples, Number of categories)
    :param predictions: torch.tensor consisting of predictive probabilities for every label: (batch_size,
    Number of samples, Number of categories)
    :param k: Value of k. Default: 1
    :param pos_label: Value to consider as positive in Multi-hot vector. Default: 1

    :return: Precision @ k for a given ground truth - prediction pair.for a batch of samples.
    """
    ## Top k probabilities
    preds_indices = torch.argsort(predictions, dim=1, descending=True)
    preds_desc = preds_indices[:, :k]

    # com_labels = []  # batch_size, Number of samples
    precision_batch = 0
    for i in np.arange(predictions.shape[0]):  # (batch_size, Number of samples, Number of categories)
        precision_samples = 0
        for j in np.arange(predictions.shape[1]):
            precision_elm = 0
            for l in np.arange(preds_desc.shape[2]):
                if actuals[i, j, preds_desc[i, j, l].item()] == pos_label:  # Checking if top index positions are 1.
                    precision_elm += 1
            precision_samples += precision_elm / preds_desc.shape[2]
        precision_batch += precision_samples / predictions.shape[1]
    precision = precision_batch / predictions.shape[0]
    return precision  # , com_labels


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


from sklearn.preprocessing import LabelBinarizer


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


def accuracy(out, labels):
    """

    :param out:
    :param labels:
    :return:
    """
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred: torch.Tensor, y_true: torch.Tensor,
                    thresh: float = 0.5, sigmoid: bool = True):
    """Compute accuracy by counting the fraction of correct predictions on
    y_pred when y_true is also True."""
    if sigmoid: y_pred = y_pred.sigmoid()

    y_pred[y_pred > thresh] = 1.
    y_pred[y_pred <= thresh] = 0.

    correct_label_count = (
            (y_pred == 1.) * (y_true.float() == 1.)).float().sum()
    total_label_count = y_true.sum()
    acc = torch.div(correct_label_count, total_label_count)
    return acc.item()


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
    logger.info(pl_dict)
    pl_sk_dict = calculate_performance_sk(true, pred)
    logger.info(pl_sk_dict)
    sk_dict = calculate_performance_sk(true.numpy(), pred.numpy())
    logger.info(sk_dict)

    true = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]])
    pred = np.array([[0, 1, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

    true = torch.from_numpy(true)
    pred = torch.from_numpy(pred)

    precision_k_hot(true, pred)


if __name__ == "__main__":
    main()
