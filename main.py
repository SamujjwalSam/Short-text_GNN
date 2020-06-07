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
from collections import OrderedDict

from config import configuration as cfg, platform as plat, username as user
from File_Handlers.csv_handler import read_tweet_csv
from File_Handlers.json_handler import save_json
from File_Handlers.pkl_handler import save_pickle
from tweet_normalizer import normalizeTweet
from build_corpus_vocab import torchtext_corpus
from Data_Handlers.torchtext_handler import dataset2iter, MultiIterator
from generate_graph import create_src_tokengraph, create_tgt_tokengraph,\
    get_k_hop_subgraph, generate_sample_subgraphs, plot_graph,\
    plot_weighted_graph, get_node_features, add_edge_weights, create_tokengraph
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


def map_nodetxt2GCNvec(G, node_list, X, return_numpy=True):
    GCNvec_dict = OrderedDict()
    for i, node in enumerate(node_list):
        node_txt = G.node[node]['node_txt']
        if return_numpy:
            GCNvec_dict[node_txt] = X[i].numpy()
        else:
            GCNvec_dict[node_txt] = X[i].numpy()

    return GCNvec_dict


def get_c_vocab(S_vocab, T_vocab, ignore_lower_freq=2):
    c_freqs = S_vocab['freqs'].copy()
    for token_str, freq in T_vocab['freqs'].items():
        try:
            c_freqs[token_str] += freq
        except KeyError:
            c_freqs[token_str] = freq

    # idx = 0
    # for token_str, freq in c_freqs.items():

    # c_freqs = S_vocab['freqs'].copy()
    c_s2i = S_vocab['str2idx_map'].copy()
    c_i2s = S_vocab['idx2str_list'].copy()
    t_idx_start = len(S_vocab['str2idx_map'])
    for token_str in T_vocab['str2idx_map'].keys():
        try:
            c_s2i[token_str]
        except KeyError:
            c_s2i[token_str] = t_idx_start
            t_idx_start += 1
            c_i2s.append(token_str)

        try:
            c_freqs[token_str] = c_freqs[token_str] + T_vocab[
                'freqs'][token_str]
        except KeyError:
            c_freqs[token_str] = T_vocab['freqs'][token_str]

    c_vocab = {
        'freqs':        c_freqs,
        'str2idx_map':  c_s2i,
        'idx2str_list': c_i2s,
    }

    return c_vocab


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
    s_unlab_df = s_unlab_df.append(s_lab_df[['text', 'domain', 'labelled']])

    S_data_name = unlabelled_source_name + "_data.csv"
    s_unlab_df.to_csv(join(data_dir, S_data_name))

    S_dataset, S_fields = torchtext_corpus(csv_dir=data_dir,
                                           csv_file=S_data_name,
                                           )

    S_vocab = {
        'freqs':        S_fields.vocab.freqs,
        'str2idx_map':  dict(S_fields.vocab.stoi),
        'idx2str_list': S_fields.vocab.itos,
    }

    # logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
    logger.info("Source vocab size: [{}]".format(len(S_fields.vocab)))

    ## Read target data
    t_unlab_df = read_tweet_csv(data_dir, unlabelled_target_name + ".csv")

    ## Prepare target data
    t_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    t_unlab_df['domain'] = 1
    t_unlab_df['labelled'] = False

    ## Target dataset
    T_data_name = unlabelled_target_name + "_data.csv"
    t_unlab_df.to_csv(join(data_dir, T_data_name))

    T_dataset, T_fields = torchtext_corpus(csv_dir=data_dir,
                                           csv_file=T_data_name,
                                           )
    logger.info("Target vocab size: [{}]".format(len(T_fields.vocab)))

    T_vocab = {
        'freqs':        T_fields.vocab.freqs,
        'str2idx_map':  dict(T_fields.vocab.stoi),
        'idx2str_list': T_fields.vocab.itos,
    }

    ## Create combined data:
    c_df = s_unlab_df.append(t_unlab_df)
    # s_unlab_df = None
    # t_unlab_df = None

    c_data_name = unlabelled_source_name + '_' + unlabelled_target_name\
                  + "_data.csv"
    c_df.to_csv(join(data_dir, c_data_name))

    c_df = None

    C_dataset, C_fields = torchtext_corpus(csv_dir=data_dir,
                                           csv_file=c_data_name,
                                           )

    c_vocab = {
        'freqs':        C_fields.vocab.freqs,
        'str2idx_map':  dict(C_fields.vocab.stoi),
        'idx2str_list': C_fields.vocab.itos,
    }

    ## Combine S and T vocabs:
    # c_vocab = get_c_vocab(S_vocab, T_vocab)
    # S_iter, T_iter = dataset2iter((S_dataset, T_dataset), batch_size=1)
    # c_iter = MultiIterator([S_iter, T_iter])
    logger.info("Combined vocab size: [{}]".format(len(c_vocab[
                                                           'str2idx_map'])))

    ## Generate embeddings for OOV tokens:
    glove_embs = glove2dict()

    ## Get all OOVs which does not have Glove embedding:
    high_oov_freqs, low_glove_freqs, corpus, corpus_toks =\
        preprocess_and_find_oov((S_dataset, T_dataset), c_vocab,
                                glove_embs=glove_embs, )

    # ## Create token graph G using source data:
    # G = create_src_tokengraph(corpus_toks[0], S_vocab)
    #
    # ## Add nodes and edges to the token graph generated using source data:
    # G, c_i2s = create_tgt_tokengraph(corpus_toks[1], T_vocab, S_vocab, G)
    G = create_tokengraph(corpus_toks, c_vocab, S_vocab, T_vocab)

    logger.info("Number of nodes in the token graph: [{}]".format(len(G.nodes)))
    logger.info("Number of edges in the token graph: [{}]".format(len(G.edges)))

    ## Create new embeddings for OOV tokens:
    high_oov_tokens_list = list(high_oov_freqs.keys())
    c_corpus = corpus[0] + corpus[1]
    oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, c_corpus)
    oov_embs = train_model(oov_mat_coo, high_oov_tokens_list, glove_embs)
    glove_embs = merge_dicts(glove_embs, oov_embs)

    ## TODO: Generate <UNK> embedding from low freq tokens:
    ## Save embedding with OOV tokens:
    # save_pickle(glove_embs, pkl_file_name=cfg["pretrain"]["pretrain_file"] +
    #                                       labelled_source_name + '_' +
    #                                       labelled_target_name,
    #             pkl_file_path=cfg["paths"]["pretrain_dir"][plat][user])

    # save_glove(glove_embs, glove_dir=cfg["paths"]["pretrain_dir"][plat][user],
    #            glove_file='oov_glove.txt')

    ## Calculate edge weights from cooccurrence stats:
    G = add_edge_weights(G)

    ## Get adjacency matrix and node embeddings in same order:
    node_list = G.nodes
    adj = nx.adjacency_matrix(G, nodelist=node_list,
                              weight='weight'
                              )
    # adj_np = nx.to_numpy_matrix(G)
    X = get_node_features(glove_embs, c_vocab['idx2str_list'], G.nodes)
    X_hat = GCN_forward(adj, X)

    ## Create text to GCN forward vectors:
    X_dict = map_nodetxt2GCNvec(G, node_list, X_hat)

    ## Save GCN forwarded vectors for future use:
    # save_pickle(X_dict, pkl_file_name='X_dict.t', pkl_file_path=data_dir)
    torch.save(X_dict, join(data_dir, 'X_dict.pt'))
    # X_dict = torch.load(join(data_dir, 'X_dict.pt'))
    save_glove(glove_embs, glove_dir=cfg["paths"]["pretrain_dir"][plat][user],
               glove_file='oov_glove.txt')

    ## Construct tweet subgraph:
    # S_iter, T_iter = dataset2iter((S_dataset, T_dataset), batch_size=1)
    # txts_subgraphs = generate_sample_subgraphs(s_lab_df.text.to_list(), G=G)
    # logger.info("Fetching subgraph: [{}]".format(txts_subgraphs))
    # print(txts_subgraphs[0].nodes)
    # plot_graph(txts_subgraphs[0])

    logger.info("Execution complete.")


def save_glove(glove_embs, glove_dir=cfg["paths"]["pretrain_dir"][plat][user],
               glove_file='oov_glove.txt'):
    with open(join(glove_dir, glove_file), 'w', encoding='UTF-8') as glove_f:
        for token, vec in glove_embs.items():
            line = token + ' ' + str(vec)
            glove_f.write(line)


if __name__ == "__main__":
    main()
