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
import pandas as pd
from os.path import join

from config import configuration as cfg, platform as plat, username as user
from read_tweets import read_tweet_csv
from tweet_normalizer import normalizeTweet
from build_corpus_vocab import build_corpus, torchtext_corpus
from generate_graph import plot_graph, generate_sample_subgraphs,\
    generate_window_token_graph_torch
from finetune_static_embeddings import glove2dict
from Logger.logger import logger


def tokenize_txts(df: pd.DataFrame, txts_toks: list = None):
    if txts_toks is None:
        txts_toks = []
    for txt in df.tweets:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    return txts_toks


def main(data_dir=cfg["paths"]["dataset_dir"][plat][user],
         labelled_source_name=cfg["data"]["source"]['labelled'],
         unlabelled_source_name=cfg["data"]["source"]['unlabelled'],
         labelled_target_name=cfg["data"]["target"]['labelled'],
         unlabelled_target_name=cfg["data"]["target"]['unlabelled'],
         ):
    ## Read source data
    s_lab_df = read_tweet_csv(data_dir, labelled_source_name + ".csv")
    s_unlab_df = read_tweet_csv(data_dir, unlabelled_source_name + ".csv",
                                )
    # pd.read_csv(open('test.csv','rU'), encoding='utf-8', engine='c')

    ## Read target data
    # t_lab_df = read_tweet_csv(data_dir, labelled_target_name+".csv")
    t_unlab_df = read_tweet_csv(data_dir, unlabelled_target_name + ".csv")

    # all_toks = []
    # s_lab_toks = tokenize_txts(s_lab_df)
    # all_toks = all_toks + s_lab_toks
    # s_unlab_toks = tokenize_txts(s_unlab_df)
    # all_toks = all_toks + s_unlab_toks
    # t_unlab_toks = tokenize_txts(t_unlab_df)
    # all_toks = all_toks + t_unlab_toks

    s_lab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_lab_df['domain'] = 0
    s_lab_df['labelled'] = True

    s_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_unlab_df['domain'] = 0
    s_unlab_df['labelled'] = False

    ## Prepare source data
    S_df = s_lab_df[['text', 'domain', 'labelled']]
    S_df = S_df.append(s_unlab_df[['text', 'domain', 'labelled']])

    S_data_name = unlabelled_source_name + "_data.csv"
    S_df.to_csv(join(data_dir, S_data_name))
    S_dataset, S_fields = torchtext_corpus(
        csv_dir=data_dir, csv_file=S_data_name)

    # logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
    logger.info("Vocab size: [{}]".format(len(S_fields.vocab.freqs)))

    ## Create token graph G using source data:
    G = generate_window_token_graph_torch(S_dataset, edge_attr='s_co')

    ## Prepare target data
    t_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    t_unlab_df['domain'] = 1
    t_unlab_df['labelled'] = False

    T_data_name = unlabelled_target_name + "_data.csv"
    t_unlab_df.to_csv(join(data_dir, T_data_name))
    T_dataset, T_fields = torchtext_corpus(
        csv_dir=data_dir, csv_file=S_data_name)

    ## Add nodes and edges to the token graph generated using source data:
    G = generate_window_token_graph_torch(T_dataset, G, edge_attr='t_co')

    logger.info("Number of nodes in the token graph: [{}]".format(len(G.nodes)))
    logger.info("Degree of nodes in the token graph: [{}]".format(G.degree))

    # txts_embs = create_node_embddings(txts_toks)

    txts_subgraphs = generate_sample_subgraphs(s_lab_df.text.to_list(), G=G)
    # logger.info("Fetching subgraph: [{}]".format(txts_subgraphs))
    # print(H.nodes)
    plot_graph(txts_subgraphs[0])
    print("Successfully printed.")


if __name__ == "__main__":
    main()
