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


def generate_token_graph(vocab: dict, corpus: list, G=None):
    if G is None:
        G = nx.Graph()

    G.add_nodes_from(vocab.items())

    for token, freq in vocab.items():
        G.add_node(token, s=freq[0], t=freq[1])

    edges = {}

    for txt in corpus:
        for i, token1 in enumerate(txt):
            for token2 in txt[i+1:]:
                # if token1 == token2:
                #     continue
                try:
                    edges[(token1, token2)] += 1
                except KeyError as e:
                    edges[(token1, token2)] = 1

    for nodes, cooccure in edges.items():
        G.add_edge(nodes[0], nodes[1], cooccure=cooccure)

    return G


def get_subgraph(G, nodes):
    H = nx.subgraph(G, nodes)

    return H
