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
from generate_graph import generate_token_graph, get_subgraph, plot_graph,\
    generate_sample_subgraphs
from finetune_static_embeddings import glove2dict
from Logger.logger import logger


def main(data_dir=cfg["paths"]["dataset_dir"][plat][user],
         data_name=cfg["data"]["dataset_name"]):
    # txts = ['This is the first sentence.',
    #         'This is the second.',
    #         'There is no sentence in this corpus longer than this one.',
    #         'My dog is named Patrick.']
    ## Read source data
    s_lab_df = read_tweet_csv(data_dir, data_name+".csv")
    s_unlab_df = read_tweet_csv(data_dir, data_name+".csv")

    ## Read target data
    t_lab_df = read_tweet_csv(data_dir, data_name+".csv")
    t_unlab_df = read_tweet_csv(data_dir, data_name+".csv")

    logger.info("Dataset size: [{}]".format(s_lab_df.shape))
    logger.info("Few dataset samples: \n[{}]".format(s_lab_df.head()))

    txts_toks = []
    for txt in s_lab_df.tweets:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    corpus, vocab = build_corpus(txts_toks)

    logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
    logger.info("Vocab size: [{}]".format(len(vocab)))

    G = generate_token_graph(vocab, txts_toks)
    logger.info("Number of nodes in the token graph: [{}]".format(len(G.nodes)))

    txts_embs = create_node_embddings(txts_toks)

    H = generate_sample_subgraphs(G, ['b', 'c'])
    logger.info("Fetching subgraph: [{}]".format(H.nodes))
    # print(H.nodes)
    plot_graph(H)
    print("Successfully printed.")

    txts_subgraphs = generate_sample_subgraphs(txts_toks, G)


if __name__ == "__main__":
    main()
