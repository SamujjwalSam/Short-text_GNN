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


def undersample_major_class(discretized_labels: dict):
    under_sampler = RandomUnderSampler()

    ## RandomSampler only takes numpy as input, convert:
    tokens = np.array(list(discretized_labels.keys())).reshape(-1, 1)
    vecs = np.array(list(discretized_labels.values()))

    undersampled_class_sets = []
    for i in range(vecs.shape[1]):
        tokens_resampled, vecs_resampled = under_sampler.fit_resample(
            tokens, vecs[:, i])
        discretized_labels_resampled = {}
        for token, vec in zip(tokens_resampled, vecs_resampled):
            token = str(token[0])
            discretized_labels_resampled[token] = vec
        undersampled_class_sets.append(discretized_labels_resampled)

    return undersampled_class_sets


def fetch_all_nodes(node_list: list, token2label_vec_map: dict,
                    token_id2token_txt_map: list, default_fill=-1.):
    """ Fetches label vectors ordered by node_list.

    :param token_id2token_txt_map:
    :param default_fill:
    :param num_classes: Number of classes
    :param node_list:
    :param token2label_vec_map: defaultdict of node to label vectors map
    :return:
    """
    all_node_embs = []
    for cls_token2label_vec_map in token2label_vec_map:
        ordered_node_embs = []
        for node in node_list:
            try:
                ordered_node_embs.append(cls_token2label_vec_map[
                                             token_id2token_txt_map[node]])
            except KeyError:
                ordered_node_embs.append(default_fill)

        all_node_embs.append(ordered_node_embs)
    all_node_embs = np.stack(all_node_embs)

    return all_node_embs


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


def propagate_multilabels(features, labels, ):
    preds = []
    for i in range(labels.shape[0]):
        pred = propagate_labels(features, labels[i, :])

        pred_argmax = np.argmax(pred, axis=1)
        test1 = np.ma.masked_where(labels[0] > -1, labels[0])
        correct = (labels[i, :][test1.mask] == pred_argmax[test1.mask]).sum()
        total = len(labels[i, :][test1.mask])
        logger.info(f'Class [{i}] accuracy: [{correct / total}]')

        preds.append(pred)

    return np.stack(preds).T


if __name__ == "__main__":
    data = prepare_features()
    labels = discretize_labels()
    preds = propagate_labels(data, labels, )
