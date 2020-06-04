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

import torch
import pandas as pd
import networkx as nx
from os.path import join
from nltk.corpus import brown

from config import configuration as cfg, platform as plat, username as user
from File_Handlers.csv_handler import read_tweet_csv
from File_Handlers.json_handler import save_json
from File_Handlers.pkl_handler import save_pickle
from tweet_normalizer import normalizeTweet
from build_corpus_vocab import torchtext_corpus
from Data_Handlers.torchtext_handler import dataset2iter, MultiIterator
from generate_graph import create_src_tokengraph, create_tgt_tokengraph,\
    get_k_hop_subgraph, generate_sample_subgraphs, plot_graph,\
    plot_weighted_graph, get_node_features
from Layers.GCN_forward import GCN_forward
from finetune_static_embeddings import glove2dict, get_rareoov, process_data,\
    calculate_cooccurrence_mat, train_model, preprocess_and_find_oov
from Logger.logger import logger


def tokenize_txts(df: pd.DataFrame, txts_toks: list = None):
    if txts_toks is None:
        txts_toks = []
    for txt in df.tweets:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    return txts_toks


def merge_dicts(*dict_args):
    """ Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts. """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def main(data_dir=cfg["paths"]["dataset_dir"][plat][user],
         labelled_source_name=cfg["data"]["source"]['labelled'],
         unlabelled_source_name=cfg["data"]["source"]['unlabelled'],
         labelled_target_name=cfg["data"]["target"]['labelled'],
         unlabelled_target_name=cfg["data"]["target"]['unlabelled'],
         ):
    ## Read source data
    s_lab_df = read_tweet_csv(data_dir, labelled_source_name + ".csv")
    s_unlab_df = read_tweet_csv(data_dir, unlabelled_source_name + ".csv")

    s_lab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_lab_df['domain'] = 0
    s_lab_df['labelled'] = True

    s_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_unlab_df['domain'] = 0
    s_unlab_df['labelled'] = False

    ## Prepare source data
    s_unlab_df = s_lab_df[['text', 'domain', 'labelled']]
    s_unlab_df = s_unlab_df.append(s_unlab_df[['text', 'domain', 'labelled']])

    S_data_name = unlabelled_source_name + "_data.csv"
    s_unlab_df.to_csv(join(data_dir, S_data_name))
    s_unlab_df = None

    S_dataset, S_fields = torchtext_corpus(csv_dir=data_dir,
                                           csv_file=S_data_name,
                                           )

    # logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
    logger.info("Source vocab size: [{}]".format(len(S_fields.vocab.freqs)))

    ## Read target data
    t_unlab_df = read_tweet_csv(data_dir, unlabelled_target_name + ".csv")

    ## Prepare target data
    t_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    t_unlab_df['domain'] = 1
    t_unlab_df['labelled'] = False

    ## Target dataset
    T_data_name = unlabelled_target_name + "_data.csv"
    t_unlab_df.to_csv(join(data_dir, T_data_name))
    t_unlab_df = None

    T_dataset, T_fields = torchtext_corpus(csv_dir=data_dir,
                                           csv_file=T_data_name,
                                           )
    logger.info("Target vocab size: [{}]".format(len(T_fields.vocab.freqs)))

    ## Combine S and T dataset iterators to find OOVs:
    # S_iter, T_iter = dataset2iter((S_dataset, T_dataset), batch_size=1)
    # combined_iter = MultiIterator([S_iter, T_iter])

    ## Generate embeddings for OOV tokens:
    glove_embs = glove2dict()

    ## Create combined dataset to get all OOVs:
    oov_vocabs, corpus = preprocess_and_find_oov((S_dataset, T_dataset),
                                                 glove_embs=glove_embs)
    coo_mat = calculate_cooccurrence_mat(oov_vocabs, corpus)

    ## Create new embeddings for OOV tokens:
    new_glove_embs = train_model(coo_mat, oov_vocabs, glove_embs)
    glove_embs = merge_dicts(glove_embs, new_glove_embs)
    ## TODO: Generate <UNK> embedding from low freq tokens:

    ## Save embedding with OOV tokens:
    save_pickle(glove_embs, pkl_file_name=cfg["pretrain"]["pretrain_file"] +
                                          labelled_source_name + '_' +
                                          labelled_target_name,
                pkl_file_path=cfg["paths"]["pretrain_dir"][plat][user])

    ## Create token graph G using source data:
    G = create_src_tokengraph(S_dataset, S_fields)

    ## Add nodes and edges to the token graph generated using source data:
    G, combined_i2s = create_tgt_tokengraph(T_dataset, T_fields, S_fields, G)

    logger.info("Number of nodes in the token graph: [{}]".format(len(G.nodes)))

    ## Calculate edge weights from cooccurrence stats:

    ## Get adjacency matrix and node embeddings in same order:
    ## TODO: Values does not contain degree instead cooccurrence
    adj = nx.adjacency_matrix(G, nodelist=G.nodes,
                              # weight='weight'
                              )
    # adj_np = nx.to_numpy_matrix(G)
    X = get_node_features(glove_embs, combined_i2s, G.nodes)
    X_hat = GCN_forward(adj, X)

    ## Construct tweet subgraph:
    txts_subgraphs = generate_sample_subgraphs(s_lab_df.text.to_list(), G=G)
    logger.info("Fetching subgraph: [{}]".format(txts_subgraphs))
    print(txts_subgraphs[0].nodes)
    plot_graph(txts_subgraphs[0])

    logger.info("Execution complete.")


if __name__ == "__main__":
    main()
