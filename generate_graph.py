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

import networkx as nx
import matplotlib.pyplot as plt


def generate_token_graph(vocab: dict, corpus: list, G:nx.Graph=None):
    """ Given a corpus create a token Graph.

    Append to graph G if provided.

    :param vocab:
    :param corpus:
    :param G:
    :return:
    """
    if G is None:
        G = nx.Graph()

    # for token, freq in vocab.items():
    #     G.add_node(token, s=freq[0], t=freq[1])

    edges = {}
    for txt in corpus:
        for i, token1 in enumerate(txt):
            for token2 in txt[i + 1:]:
                ## Should we create self loop if a word occurs in 2 places in
                # a single tweet?
                # if token1 == token2:
                #     continue
                try:
                    edges[(token1, token2)] += 1
                except KeyError as e:
                    edges[(token1, token2)] = 1

    for nodes, cooccure in edges.items():
        G.add_edge(nodes[0], nodes[1], cooccure=cooccure)

    return G


def get_subgraph(G:nx.Graph, nodes:list):
    H = nx.subgraph(G, nodes)

    return H


def plot_graph(G:nx.Graph):
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    # plt.subplot(122)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()
    plt.show()
    plt.savefig("H.png")


def add_sample_edges(txt:list, H:nx.Graph):
    """ For a given input text sample and it's neighbors in token graph.
    Connect nodes present in sample text with edge and calculate edge weight.

    :param txt:
    :param H:
    """
    pass


def generate_sample_subgraph(txts:list, G:nx.Graph):
    """ Given a sample texts, generate subgraph keeping the sample texts
     connected.

    :param txts:
    :param G:
    """
    txts_subgraphs = {}
    for i, txt in enumerate(txts):
        # nodes_list = []
        # for token in txt:
        txts_subgraphs[i] = G.subgraph(txt)
        txts_subgraphs[i] = add_sample_edges(txt, txts_subgraphs[i])


if __name__ == "__main__":
    txts = ['This is the first sentence.',
            'This is the second.',
            'There is no sentence in this corpus longer than this one.',
            'My dog is named Patrick.']

    from tweet_normalizer import normalizeTweet

    txts_toks = []
    for txt in txts:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    from build_corpus_vocab import build_corpus

    corpus, vocab = build_corpus(txts_toks)

    G = generate_token_graph(vocab, txts_toks)
    print(G.nodes)
    H = get_subgraph(G, ['This', 'is'])
    print(H.nodes)
    plot_graph(H)
    print("Successfully printed.")
