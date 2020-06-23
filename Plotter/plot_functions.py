# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__sdiables__   :
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

import statistics
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from Logger.logger import logger

plt.rcParams.update({'font.size': 15})


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


def plot_training_loss(training_losses: list, val_losses: list,
                       plot_name: str='training_loss'):
    """ Plots loss comparison of training and validation.

    :param plot_name:
    :param training_losses:
    :param val_losses:
    """
    # Create count of the number of epochs
    epoch_count = range(1, len(training_losses) + 1)

    fig, ax = plt.subplots()

    # Visualize loss history
    plt.plot(epoch_count, training_losses, 'r--')
    plt.plot(epoch_count, val_losses, 'b-')
    plt.title('Epoch-wise training and validation loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.show()
    fig.savefig(plot_name+'.pdf', format='pdf', bbox_inches='tight')


def calculate_sd():
    for name, dataset in {"FIRE16":FIRE16, "SMERP17":smerp_results,
                          "Kaggle":kaggle_results}.items():
        print(f"Dataset: [{name}]")
    for algo, data in dataset.items():
        print(f"\tMetric: [{algo}]")
        for i, v in data.items():
            sd = statistics.stdev(v)
            print(f"\t\t {sd:.3},")


def plot_pr(data, classes=None, n_groups=7, bar_elev=.4, set_ylim=0.35,
            dataset='FIRE16', ylabel=None, error_kw=None):
    """ Plots bar for result comarison from mean and variance.

    :param data: 4-tuple: list of mean scores and variance.
    :param classes:
    :param n_groups:
    :param bar_elev:
    :param set_ylim:
    :param dataset:
    :param ylabel:
    :param error_kw:
    """
    if error_kw is None:
        error_kw = {"ecolor": "black", "elinewidth": 10, }

    baseline, approach, baseline_sd, approach_sd = data

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    rects1 = plt.bar(index, baseline, bar_width,
                     alpha=opacity,
                     color='white',
                     label='Sup.',
                     hatch="////",
                     edgecolor='black',
                     yerr=baseline_sd,
                     error_kw=error_kw
                     )

    rects2 = plt.bar(index + bar_width + 0.05, approach, bar_width,
                     alpha=opacity,
                     color='white',
                     label='SSF',
                     # hatch='o.'
                     edgecolor='black',
                     yerr=approach_sd,
                     error_kw=error_kw
                     )

    plt.xlabel('Classes')
    plt.ylabel(ylabel)

    plt.title('Class-wise scores of ' + dataset)
    if classes is None:
        if dataset == 'FIRE16':
            classes = (
                '1 (271)', '2 (149)', '3 (160)', '4 (52)', '5 (96)', '6 (181)',
                '7 (108)')
        elif dataset == 'SMERP17':
            classes = ('1 (174)', '2 (119)', '3 (1129)', '4 (202)')
        elif dataset == 'Kaggle':
            classes = ('0 (4342)', '1 (3271)')
        else:
            raise NotImplementedError(f'Class details (x-axis) not found '
                                      f'for [{dataset}]')

    plt.xticks(index + bar_width, classes)
    plt.legend(loc='upper left')

    plt.tight_layout()
    ax.set_ylim(bottom=set_ylim)

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1] * 2, forward=True)

    for bar in rects1:
        yval = bar.get_height()
        plt.text(bar.get_x() - bar_elev, yval + .02, yval)

    for bar in rects2:
        yval = bar.get_height()
        plt.text(bar.get_x() + .005, yval + .03, yval)

    # plt.show()
    fig.savefig(dataset + ylabel + '.eps', format='eps', bbox_inches='tight')


if __name__ == '__main__':
    import numpy as np

    X = np.random.random((500, 100))
    plot_features_tsne(X)

    fire = {
        'Precision': [(0.7342, 0.7021, 0.8020, 0.4130, 0.5194, 0.5473, 0.9132),
                      (0.7312, 0.7114, 0.8033, 0.4077, 0.5151, 0.5479, 0.8987),
                      (0.0139, 0.0140, 0.0108, 0.0252, 0.0317, 0.0182, 0.0088,),
                      (0.0189, 0.0180, 0.0130, 0.0277, 0.0247, 0.0102, 0.0108)],
        'Recall':    [(0.7216, 0.6933, 0.7702, 0.0821, 0.2152, 0.5314, 0.9147),
                      (0.7224, 0.7379, 0.7751, 0.1944, 0.2506, 0.5357, 0.9027),
                      (0.0107, 0.0200, 0.0100, 0.0304, 0.0226, 0.0151, 0.0090),
                      (0.0147, 0.0167, 0.0150, 0.0400, 0.0306, 0.0211, 0.0159)]
    }

    smerp = {
        'Precision': [(0.7513, 0.8116, 0.9884, 0.8838),
                      (0.7538, 0.8220, 0.9891, 0.8842),
                      (0.0089, 0.0110, 0.0070, 0.0087),
                      (0.0111, 0.0160, 0.0089, 0.0107)
                      ],
        'Recall':    [(0.7489, 0.7731, 0.9921, 0.8708),
                      (0.7587, 0.8363, 0.9942, 0.8908),
                      (0.0107, 0.0217, 0.005, 0.0104),
                      (0.0147, 0.0180, 0.0075, 0.0154)
                      ],
    }

    sup_ebar = {
        "ecolor":     "black",
        "elinewidth": 6,
        # "barsabove": True
    }

    for metric, data in fire.items():
        plot_pr(data, n_groups=7, bar_elev=.35, set_ylim=0.0, dataset='FIRE16',
                ylabel=metric, error_kw=sup_ebar)

    sup_ebar = {
        "ecolor":     "black",
        "elinewidth": 15,
        # "barsabove": True
    }

    for metric, data in smerp.items():
        plot_pr(data, n_groups=4, bar_elev=.1, set_ylim=0.65, dataset='SMERP17',
                ylabel=metric, error_kw=sup_ebar)
