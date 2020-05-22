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

import argparse

from config import configuration as cfg, platform as plat, username as user
from read_tweets import read_tweet_csv
from tweet_normalizer import normalizeTweet
from build_corpus_vocab import build_corpus
from generate_graph import generate_token_graph, get_subgraph, plot_graph


def main(data_dir=cfg["paths"]["dataset_dir"][plat][user],
         data_name=cfg["data"]["dataset_name"]):
    # txts = ['This is the first sentence.',
    #         'This is the second.',
    #         'There is no sentence in this corpus longer than this one.',
    #         'My dog is named Patrick.']
    txts_df = read_tweet_csv(data_dir, data_name+".csv")

    txts_toks = []
    for txt in txts_df.tweets:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    corpus, vocab = build_corpus(txts_toks)

    G = generate_token_graph(vocab, txts_toks)
    print(G.nodes)
    H = get_subgraph(G, ['This', 'is'])
    print(H.nodes)
    plot_graph(H)
    print("Successfully printed.")


if __name__ == "__main__":
    main()
