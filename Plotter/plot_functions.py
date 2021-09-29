# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
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

import torch
# import umap
import statistics
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from os.path import join
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from config import dataset_dir, pretrain_dir
from Logger.logger import logger

plt.rcParams.update({'font.size': 20})
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.color_palette("viridis", as_cmap=True)


def plot_umap(data, save_name='umap_vecs.pdf'):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette()[x] for x in data])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection:', fontsize=24)
    plt.savefig(save_name)
    plt.show()


def plot_vecs_color(tokens2vec, axis_range=None, save_name='tsne_vecs.pdf',
                    title='TSNE visualization of top 10 tokens in 2D'):
    """ Plots TSNE representations of tokens and their embeddings.

    :param tokens2vec:
    :param axis_range:
    :param save_name:
    :param title:
    :param X:
    :param tokens:
    :param limit_view:
    """
    tsne = TSNE(n_components=2)

    X = np.stack(list(tokens2vec.values()))
    tokens = list(tokens2vec.keys())

    X_2d = tsne.fit_transform(X)
    # colors = axis_range(X_2d.shape[0])
    # print(len(colors), colors)

    plt.figure(figsize=(15, 15))
    if tokens is not None:
        for i, token in enumerate(tokens):
            plt.annotate(token, xy=(X_2d[i, 0], X_2d[i, 1]), zorder=1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=60, alpha=.5,  # c=colors
                )
    if axis_range is not None:
        plt.xlim(-axis_range, axis_range)
        plt.ylim(-axis_range, axis_range)
    plt.tight_layout()
    plt.title(title)
    # plt.xlabel('x-axis')
    # plt.ylabel('y-axis')
    plt.savefig(save_name)
    plt.show()


def plot_graph(g):
    plt.subplot(122)
    nx.draw(g.to_networkx(), with_labels=True)

    plt.show()


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
                       plot_name: str = 'training_loss'):
    """ Plots loss comparison of training and validation.

    :param plot_name:
    :param training_losses:
    :param val_losses:
    """
    # Create count of the number of epoch
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
    fig.savefig(plot_name + '.pdf', format='pdf', bbox_inches='tight')


def calculate_sd():
    for name, dataset in {"FIRE16": FIRE16, "SMERP17": smerp_results,
                          "Kaggle": kaggle_results}.items():
        logger.info(f"Dataset: [{name}]")
    for algo, data in dataset.items():
        logger.info(f"\tMetric: [{algo}]")
        for i, v in data.items():
            sd = statistics.stdev(v)
            logger.info(f"\t\t {sd:.3},")


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


def plot_weighted_graph(G, val=2):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["s_co"] > val]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["s_co"] <= val]

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b",
        style="dashed"
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    plt.axis("off")
    plt.show()


def plot_graph(G: nx.Graph, plot_name: str = 'H.png', labels: dict = None):
    """ Plots a networkx graph.

    :param G:
    :param plot_name:
    :param labels: Node labels map from id to text.
    """
    plt.subplot(121)
    # labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    if labels:
        nx.draw(G, labels=labels, with_labels=True, font_weight='bold')
    else:
        nx.draw(G, with_labels=True, font_weight='bold')
    # plt.subplot(122)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()
    plt.show()
    plt.savefig(plot_name)


def plot_occurance(losses: list, title="Losses", ylabel="Loss", xlabel="Epoch", clear=True, log_scale=False,
                   plot_name=None,
                   plot_dir="", show_plot=False):
    """ Plots the validation loss against epoch.

    :param plot_name:
    :param plot_dir:
    :param xlabel:
    :param ylabel:
    :param title:
    :param losses:
    :param clear:
    :param log_scale:
    """
    ## Turn interactive plotting off
    plt.ioff()

    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel(xlabel)
    if log_scale:
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.title(title)
    if plot_name is None: plot_name = title + "_" + ylabel + "_" + xlabel + ".jpg"
    plt.savefig(join(plot_dir, plot_name))
    logger.info(f"Saved plot with title [{title}] and ylabel [{ylabel}] and "
                f"xlabel [{xlabel}] at [{join(plot_dir, plot_name)}].")
    if clear:
        plt.cla()
    if show_plot: plt.show()
    plt.close(fig)  # Closing the figure so it won't get displayed in console.


def plot_heatmap(data, vmin=-1., vmax=1., save_name='heatmap.pdf'):
    """ Plots a heatmap.

    :param data: DataFrame or ndarray.
    :param vmin:
    :param vmax:
    :param save_name:
    """

    fig, ax = plt.subplots()
    plt.figure(figsize=(11.7, 8.27))
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax, annot=False, fmt="f",
                     linewidths=.5, cbar=True, cmap='viridis_r')
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()


from os.path import exists
from torch import load
from Utils.utils import get_token2pretrained_embs
from File_Handlers.json_handler import save_json, read_json


def gcpt_heatmap_glove_vs_gcpt(pretrainedX_path, token_embs_path, vocab_path):

    if exists(pretrainedX_path):
        logger.info(f'Loading Pretraining Artifacts from [{token_embs_path}]')
        vocab = read_json(vocab_path)

        X = load(pretrainedX_path)

        if exists(token_embs_path):
            token2pretrained_embs = load(token_embs_path)
        else:
            token2pretrained_embs = get_token2pretrained_embs(
                X, list(vocab['str2idx_map'].keys()), vocab['idx2str_map'])


def test_plot_heatmap(token_embs_path=join(pretrain_dir, 'pretrained', 'token2pretrained_' + str(80) + '.pt')):
    from Text_Encoder.finetune_static_embeddings import glove2dict

    token2pretrained_embs = load(token_embs_path)
    glove_embs = glove2dict()

    common = []
    # common = ['common', 'works', 'fear', 'system', 'honestly', 'such', 'trapped', 'technology', 'collect', 'thoughts',
    #           'rise', 'hours', 'dollars']
    # common.extend(common)
    dataset_tokens_set = set(token2pretrained_embs.keys())
    glove_tokens_set = set(glove_embs.keys())
    common = dataset_tokens_set.intersection(glove_tokens_set)

    dataset_name = pretrain_dir.split("/")[-1]
    joint_vocab = read_json(join(pretrain_dir, dataset_name+'_joint_vocab'))
    pos_vocab = read_json(join(pretrain_dir, dataset_name+'_pos_vocab'))
    neg_vocab = read_json(join(pretrain_dir, dataset_name+'_neg_vocab'))

    selected = ['corporation', 'unclear', 'filling', 'additional', 'pledges',
                'slide', 'reversing', 'particularly', 'cared', 'miraculously',
                'properly', 'recovering']

    # pos_exp = ['corporation', 'unclear', 'filling', 'additional', 'pledges', 'slide', 'reversing',
    #            'particularly', 'cared', 'miraculously', 'properly', 'recovering']
    # common.extend(pos_exp)
    #
    # neg_exp = ['successful', 'sounding', 'equally', 'antique', 'beautifully', 'stink', 'sauce', 'homecoming',
    #            'emotions', 'lick', 'atheist', 'fancy', 'coconut']
    # common.extend(neg_exp)

    # glove_tensor = {word: glove_embs[word] for word in common}
    # gcn_tensor = {word: token2pretrained_embs[word] for word in common}

    glove_tensor = []
    gcn_tensor = []
    words = []

    for word in selected:
        if word in glove_embs and word in token2pretrained_embs:
            glove_tensor += [glove_embs[word].tolist()]
            gcn_tensor += [token2pretrained_embs[word].tolist()]
            words.append(word)

    gcn_sim = cosine_similarity(gcn_tensor)
    gcn_sim = (gcn_sim - gcn_sim.mean()) / gcn_sim.std()

    glove_sim = cosine_similarity(glove_tensor)
    glove_sim = (glove_sim - glove_sim.mean()) / glove_sim.std()

    # index = sorted(words)
    # columns = sorted(words)

    # gcn_sigm = torch.sigmoid(torch.from_numpy(gcn_normed)).numpy()
    gcn_df = pd.DataFrame(gcn_sim, index=words, columns=words)
    gcn_df.to_csv('gcpt_sim_df.csv')
    plot_heatmap(data=gcn_df, save_name='gcpt_heatmap.pdf')

    # glove_sigm = torch.sigmoid(torch.from_numpy(glove_normed)).numpy()
    glove_df = pd.DataFrame(glove_sim, index=words, columns=words)
    glove_df.to_csv('glove_sim_df.csv')
    plot_heatmap(data=glove_df, save_name='glove_heatmap.pdf')

    save(Tensor(gcn_sim), 'gcpt_cosine_sim.pt')
    save(Tensor(glove_sim), 'glove_cosine_sim.pt')


if __name__ == '__main__':
    test_plot_heatmap()

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
