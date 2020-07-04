# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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
from sklearn.semi_supervised import LabelSpreading
from sklearn.multiclass import OneVsRestClassifier


def discretize_labels(x_vectors: np.ndarray, thresh=0.01, k = 2,):
    """ Converts logit to multi-hot based on threshold per class.

    :param x_vectors:
    :param predictions_df:
    :param cls_thresh:
    :param drop_irrelevant: Remove samples for which no class crossed it's
    threshold. i.e. [0,0,0,0]
    """
    # ## value greater than threshold:
    # x_vectors[(x_vectors > 0.0) & (x_vectors <= thresh)] = 0.0
    # x_vectors[(x_vectors > thresh)] = 1.0
    #
    # ## Maximum of each row is 1.:
    # x_vectors = (x_vectors == x_vectors.max(axis=1)[:,None]).astype(
    #     int).astype(float)

    ## Top k values of each row:
    for row in x_vectors:
        if row.sum() > 0.:  ## for non -1 rows only
            # row_tmp = np.argpartition(-row, k)
            row[np.argpartition(-row, k)[:k]] = 1.
            row[np.argpartition(-row, k)[k:]] = 0.

    return x_vectors


def construct_graph(input):
    adj = sparse.load_npz("adj.npz")
    return adj


def propagate_labels(features, labels, ):
    label_prop_model = OneVsRestClassifier(LabelSpreading(kernel=construct_graph, n_jobs=-1))

    label_prop_model.fit(features, labels)

    preds = label_prop_model.predict(features)

    return preds


if __name__ == "__main__":
    data = prepare_features()
    labels = discretize_labels()
    preds = propagate_labels(data, labels, )
