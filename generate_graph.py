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

from Logger.logger import logger


def plot_graph(G: nx.Graph, plot_name='H.png'):
    plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    # plt.subplot(122)
    # nx.draw_shell(G, with_labels=True, font_weight='bold')
    # plt.show()
    plt.show()
    plt.savefig(plot_name)


def find_cooccurrences(txt_window):
    edges = {}
    for i, token1 in enumerate(txt_window):
        for token2 in txt_window[i + 1:]:
            try:
                edges[(token1, token2)] += 1
            except KeyError as e:
                edges[(token1, token2)] = 1

    return edges


def generate_token_graph_window(corpus: list, G: nx.Graph = None,
                                window_size: int = 2):
    """ Given a corpus create a token Graph.

    Append to graph G if provided.

    :param window_size: Sliding window size
    :param corpus: List of list of str.
    :param G:
    :return:
    """
    if G is None:
        G = nx.Graph()

    # for token, freq in vocab.items():
    #     G.add_node(token, s=freq[0], t=freq[1])

    sample_edges = {}
    for i, txt in enumerate(corpus):
        j = 0
        sample_edges[i] = []
        txt_len = len(txt)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        for k in range(slide):
            txt_window = txt[j:j + window_size]
            ## Co-occurrence in tweet:
            sample_edges[i].append(find_cooccurrences(txt_window))
            j = j + 1
            # j = j + window_size-1

    for cooccure in sample_edges.values():
        for edge in cooccure:
            for nodes in edge.keys():
                G.add_edge(nodes[0], nodes[1], cooccure=cooccure)

    return G


def get_k_hop_subgraph(G: nx.Graph, txt: list,
                       # edge_attr: str = 'cooccure',
                       s_weight: float = 1.,
                       ):
    oov_nodes = []
    all_neighbors = []
    for pos, token in enumerate(txt):
        try:
            for tok in G.neighbors(token):
                all_neighbors.append(tok)
        except nx.exception.NetworkXError or nx.exception.NodeNotFound:
            logger.warn(f"Token [{token}] not present in graph.")
            oov_nodes.append((pos, token))

    G_sub = gen_sample_subgraph(all_neighbors, G)

    if oov_nodes:
        ## Add oov tokens to graph and connect it to other nodes.
        for pos, token in oov_nodes:
            if pos == 0:  ## if first token is oov
                G_sub.add_edge(txt[pos], txt[pos+1], weight=s_weight)
            elif pos == len(txt):  ## if last token if oov
                G_sub.add_edge(txt[pos-1], txt[pos], weight=s_weight)
            else:  ## Connect to previous and next node with oov node
                G_sub.add_edge(txt[pos-1], txt[pos], weight=s_weight)
                G_sub.add_edge(txt[pos], txt[pos+1], weight=s_weight)

    return G_sub


def ego_graph_nbunch_window(G: nx.Graph, nbunch: list,
                            edge_attr: str = 'cooccure',
                            s_weight: float = 1.,
                            ):
    """ Ego_graph for a bunch of nodes, adds edges among them. connects nodes
     in nbunch with original edge weight if exists, s_weight if not.

     Here, window_size is always 2.

    :param G:
    :param nbunch:
    :param edge_attr:
    :param s_weight:
    :return:
    """

    if len(nbunch) == 1:
        combine = nx.ego_graph(G, nbunch[0])
    else:
        combine = None
        for i in range(len(nbunch)):
            try:
                # node1 = nx.ego_graph(G, nbunch[i-1])
                node1 = nx.ego_graph(G, nbunch[i])
                try:  ## Merge 2 ego graphs
                    if combine is None:  ## New merged graph
                        combine = node1
                    else:  ## Merge with existing graph
                        combine = nx.compose(combine, node1)
                        try:
                            ## Copy edge weight if exists
                            combine[nbunch[i - 1]][nbunch[i]][edge_attr] =\
                                G[nbunch[i - 1]][nbunch[i]][edge_attr]
                        except KeyError as e:
                            ## If edge not exist, add edge with [s_weight].
                            combine.add_edge(nbunch[i], nbunch[i - 1],
                                             edge_attr=s_weight)
                            # combine[node1][node2][edge_attr] = s_weight
                except KeyError as e:
                    continue
            except nx.exception.NodeNotFound as e:
                continue
                ## TODO: Ignoring non-existing nodes for now; need to handle
                ## similar to OOV token
                # print(f"Node [{nbunch[i]}] not found in G.")
                # if i > 0:
                #     G.add_edge(nbunch[i-1], nbunch[i], cooccure=s_weight)
                #     G.add_edge(nbunch[i], nbunch[i+1], cooccure=s_weight)
                # else:
                #     G.add_edge(nbunch[i], nbunch[i+1], cooccure=s_weight)

    return combine


def gen_sample_subgraph(txt: list, H: nx.Graph, s_weight: float = 1.0):
    # H = G.subgraph(txt)
    for i, token1 in enumerate(txt):
        for token2 in txt[i + 1:]:
            H.add_edge(token1, token2, cooccure=s_weight)
    return H


def generate_sample_subgraphs(txts: list, G: nx.Graph, s_weight: float = 1.):
    """ Given a sample texts, generate subgraph keeping the sample texts
     connected.

    :param s_weight: Weight for edges in sample text.
    :param txts: List of texts, each containing list of tokens(nodes).
    :param G: Token graph
    """
    txts_subgraphs = {}
    for i, txt in enumerate(txts):
        # H = ego_graph_nbunch_window(G, txt)
        H = get_k_hop_subgraph(G, txt)

        # for i, txt in enumerate(txts):
        #     H = G.subgraph(txt)
        #     ## Add sample edges
        #     H = gen_sample_subgraph(txt, H, s_weight)
        #     # for i, token1 in enumerate(txt):
        #     #     for token2 in txt[i + 1:]:
        #     #         H.add_edge(token1, token2, cooccure=s_weight)

        txts_subgraphs[i] = H
    return txts_subgraphs


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

    G = generate_token_graph_window(txts_toks)
    print(G.nodes)

    ## Testing
    test_txts = ['There sam is no sentence',
                 'My dog is named sam.']
    test_txts_toks = []
    for txt in test_txts:
        test_txts_toks.append(normalizeTweet(txt, return_tokens=True))
    # txt = ['dog', 'first', 'sam']
    # H = G.subgraph(txt).copy()
    # H = nx.node_connected_component(G, txt)
    # H = nx.node_connected_component(G, txt).copy()
    txt_h = generate_sample_subgraphs(test_txts_toks, G)
    # txt_h = ego_graph_nbunch(G, txt)

    for txt in txt_h.values():
        print(txt.nodes)
        plot_graph(txt)
    print("Successfully printed.")
