# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Short summary of the script.
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

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_features_tsne(X, tokens: list = None, limit_view: int = -100):
    """ Plots TSNE representations of tokens and their embeddings.

    :param X:
    :param tokens:
    :param limit_view:
    """
    tsne = TSNE(n_components=2, random_state=0)

    if limit_view > X.shape[0]:
        limit_view = X.shape[0]

    if limit_view > 0:
        X = X[:limit_view, ]
        tokens = tokens[:limit_view]
    elif limit_view < 0:
        X = X[limit_view:, ]
        tokens = tokens[limit_view:]
    else:
        pass

    X_2d = tsne.fit_transform(X)
    colors = range(X_2d.shape[0])

    plt.figure(figsize=(6, 5))
    if tokens is not None:
        for i, token in enumerate(tokens):
            plt.annotate(token, xy=(X_2d[i, 0], X_2d[i, 1]), zorder=1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=60, alpha=.5)
    plt.title('TSNE visualization of input vectors in 2D')
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    plt.show()


def plot_training_loss(training_losses: list, val_losses: list):
    """ Plots loss comparison of training and validation.

    :param training_losses:
    :param val_losses:
    """
    # Create count of the number of epochs
    epoch_count = range(1, len(training_losses) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_losses, 'r--')
    plt.plot(epoch_count, val_losses, 'b-')
    plt.title('Epoch-wise training and validation loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    import numpy as np

    X = np.random.random((500, 100))
    plot_features_tsne(X)
