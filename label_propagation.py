# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Generate token graph
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import numpy as np
from scipy import sparse
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from imblearn.under_sampling import RandomUnderSampler

from Logger.logger import logger


def discretize_labels(x_vectors: np.ndarray, thresh1=0.2, thresh2=0.1, k=2,
                      label_neg=0.):
    """ Converts logit to multi-hot based on threshold per class.

    :param x_vectors:
    :param label_neg: Value to assign when no class should be assigned [0., -1.]
    """
    # ## value greater than threshold:
    # x_vectors[(x_vectors > 0.0) & (x_vectors <= thresh)] = 0.0
    x_vectors[(x_vectors > thresh1)] = 1.
    x_vectors[(thresh2 < x_vectors) & (x_vectors <= thresh1)] = 1.
    #
    # ## Maximum of each row is 1.:
    # x_vectors = (x_vectors == x_vectors.max(axis=1)[:,None]).astype(
    #     int).astype(float)

    ## Top k values of each row:
    for row in x_vectors:
        # for i, val in enumerate(row):
        row_sum = row.sum()
        if 0.5 < row_sum <= 1.:  ## for non -1 rows only
            row_idx = np.argpartition(-row, k)
            row[row_idx[:k]] = 1.
            row[row_idx[k:]] = label_neg
        elif 0. <= row_sum <= 0.5:
            row_idx = np.argmax(row)
            row[row_idx] = 1.
            row[(row != 1.0)] = label_neg
        elif row_sum > 1.:
            row[(row < 1.0)] = label_neg

    return x_vectors


def discretize_labelled(labelled_dict: dict, thresh1=0.1, k=2,
                        label_neg=0., label_pos=1.):
    """ Discretize probabilities of labelled tokens.

    :param labelled_dict: token:np.array(vector)
    :param label_neg: Value to assign when no class should be assigned [0., -1.]
    """
    logger.info(f'Discretizing label vector with threshold [{thresh1}].')
    labelled_vecs = np.array(list(labelled_dict.values()))

    ## value greater than threshold:
    labelled_vecs[(labelled_vecs > thresh1)] = 1.
    # labelled_vecs[(thresh2 < labelled_vecs) & (labelled_vecs <= thresh1)] = 1.

    discretized_dict = {}
    ## Top k values of each row:
    for token, vec in zip(labelled_dict.keys(), labelled_vecs):
        row_sum = vec.sum()
        if 0.5 < row_sum <= 1.:  ## for non -1 rows only
            row_idx = np.argpartition(-vec, k)
            vec[row_idx[:k]] = label_pos
            vec[row_idx[k:]] = label_neg
        elif 0. <= row_sum <= 0.5:
            row_idx = np.argmax(vec)
            vec[row_idx] = label_pos
            vec[(vec != 1.0)] = label_neg
        elif row_sum > 1.:
            vec[(vec < 1.0)] = label_neg

        discretized_dict[token] = vec

    return discretized_dict


def unison_shuffled_copies(a: np.ndarray, b: np.ndarray):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def undersample_major_class(X: np.ndarray, Y: np.ndarray, k=3):
    """ Undersamples the majority class k times.

    :param X:
    :param Y:
    :param k:
    :return:
    """
    logger.info(f'Undersampling the majority class [{k}] times.')
    under_sampler = RandomUnderSampler()
    k_undersampled_list = []
    for i in range(k):
        X_resampled, Y_resampled = under_sampler.fit_resample(X, Y)
        X_resampled, Y_resampled = unison_shuffled_copies(X_resampled,
                                                          Y_resampled)
        undersampled_dict = {}
        for x, y in zip(X_resampled, Y_resampled):
            x = str(x[0])
            undersampled_dict[x] = y

        k_undersampled_list.append(undersampled_dict)

    return k_undersampled_list


def undersample_major_multiclass(discretized_labels: dict, k=3):
    """ Undersamples the majority class for each class.

    :param discretized_labels:
    :param k:
    :return: List of dict with token to label map
    """
    ## RandomSampler only takes numpy as input; converting:
    tokens = np.array(list(discretized_labels.keys())).reshape(-1, 1)
    vecs = np.array(list(discretized_labels.values()))

    undersampled_class_sets = []
    for i in range(vecs.shape[1]):
        resampled = undersample_major_class(tokens, vecs[:, i], k=k)
        undersampled_class_sets.append(resampled)

    return undersampled_class_sets


def fetch_all_cls_nodes(node_list: list, token2label_vec_map: dict,
                        token_id2token_txt_map: list, default_fill=-1.):
    """ Fetches label vectors ordered by node_list for all classes.

    :param token_id2token_txt_map:
    :param default_fill:
    :param num_classes: Number of classes
    :param node_list:
    :param token2label_vec_map: dict of node to label vectors map
    :return:
    """
    all_node_embs = []
    for cls_token2label_vec_map in token2label_vec_map:
        us_embs = []
        for us_labels in cls_token2label_vec_map:
            cls_node_embs = fetch_all_nodes(
                node_list, us_labels, token_id2token_txt_map, default_fill)
            us_embs.append(cls_node_embs)
        all_node_embs.append(us_embs)

    return all_node_embs


def fetch_all_nodes(node_list: list, token2label_vec_map: dict,
                    token_id2token_txt_map: list, default_fill=-1.):
    """ Fetches label vectors ordered by node_list.

    :param token_id2token_txt_map:
    :param default_fill:
    :param num_classes: Number of classes
    :param node_list:
    :param token2label_vec_map: dict of node to label vectors map
    :return:
    """
    ordered_node_embs = []
    for node in node_list:
        try:
            ordered_node_embs.append(token2label_vec_map[
                                         token_id2token_txt_map[node]])
        except KeyError:
            ordered_node_embs.append(default_fill)

    return ordered_node_embs


def construct_graph(input1, input2):
    adj = sparse.load_npz("adj.npz")
    return adj


def propagate_labels(features, labels, ):
    label_prop_model = LabelSpreading(kernel=construct_graph, n_jobs=-1)
    label_prop_model.fit(features, labels)
    logger.debug(label_prop_model.classes_)
    # preds = label_prop_model.predict(features)
    preds = label_prop_model.predict_proba(features)
    # logger.debug(label_prop_model.classes_)

    return preds


def majority_voting(preds_set):
    logger.info("Taking majority voting.")
    if isinstance(preds_set, list):
        majority_count = (len(preds_set) // 2) + 1
    elif isinstance(preds_set, np.ndarray):
        majority_count = (preds_set.shape[0] // 2) + 1
    else:
        NotImplementedError(f"datatype {type(preds_set)} not supported.")

    pred_major = []
    for pred in preds_set:
        pred_discreet = np.argmax(pred, axis=1)
        pred_major.append(pred_discreet)

    pred_major = np.sum(pred_major, axis=0)

    pred_major[(pred_major < majority_count)] = 0.
    pred_major[(pred_major >= majority_count)] = 1.

    return pred_major


def lpa_accuracy(preds_set, labels):
    labels = np.stack(labels)
    # pred_argmax = []
    result = {}
    for cls in range(preds_set.shape[1]):
        # pred_argmax.append(np.argmax(pred, axis=1))
        logger.info(f'Calculating accuracy for class: [{cls}]')
        test1 = np.ma.masked_where(labels[:, cls] > 0, labels[:, cls])
        correct = (labels[:, cls][test1.mask] ==
                   preds_set[:, cls][test1.mask]).sum()
        total = labels[test1.mask].shape[0]

        result[cls] = (correct, total, correct / total)
        logger.info(f'Accuracy class: [{correct / total, correct, total}]')

    return result


def propagate_multilabels(features, labels, ):
    all_preds = []
    for i, labels_cls in enumerate(labels):
        logger.info(f'Propagating labels for class [{i}].')
        preds = []
        for under_set in labels_cls:
            pred = propagate_labels(features, np.stack(under_set))
            preds.append(pred)
        voted_preds = majority_voting(preds)
        all_preds.append(voted_preds)

    return np.stack(all_preds).T


if __name__ == "__main__":
    data = prepare_features()
    labels = discretize_labels()
    preds = propagate_labels(data, labels, )
