# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
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
import random
import argparse
import numpy as np
from torch import cuda, save, load
from torch.utils.data import DataLoader
from pandas import DataFrame
from networkx import adjacency_matrix
from os import environ
from os.path import join, exists
from json import dumps

from Label_Propagation_PyTorch.label_propagation import fetch_all_nodes, label_propagation
from Utils.utils import count_parameters, logit2label, freq_tokens_per_class, split_target, sp_coo2torch_coo
from Layers.BiLSTM_Classifier import BiLSTM_Classifier
from File_Handlers.csv_handler import read_csv, load_csvs
from File_Handlers.json_handler import save_json, read_json, read_labelled_json
from File_Handlers.read_datasets import load_fire16, load_smerp17
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Data_Handlers.torchtext_handler import dataset2bucket_iter
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Data_Handlers.instance_handler_dgl import Instance_Dataset_DGL
# from Layers.GCN_forward import GCN_forward_old
from Trainer.graph_trainer import graph_multilabel_classification
from Text_Encoder.finetune_static_embeddings import glove2dict, calculate_cooccurrence_mat,\
    train_mittens, preprocess_and_find_oov
from Trainer.trainer import trainer, predict_with_label
from Plotter.plot_functions import plot_training_loss
from Metrics.metrics import calculate_performance_pl
from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Logger.logger import logger

## Enable multi GPU cuda environment:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def set_all_seeds(seed=0):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_all_seeds(0)

"""
In [1]: import torch

In [2]: torch.cuda.current_device()
Out[2]: 0

In [3]: torch.cuda.device(0)
Out[3]: <torch.cuda.device at 0x7efce0b03be0>

In [4]: torch.cuda.device_count()
Out[4]: 1

In [5]: torch.cuda.get_device_name(0)
Out[5]: 'GeForce GTX 950M'

In [6]: torch.cuda.is_available()
Out[6]: True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Using device:', device)
logger.info()

#Additional Info when using cuda
if device.type == 'cuda':
    logger.info(torch.cuda.get_device_name(0))
    logger.info('Memory Usage:')
    logger.info('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    logger.info('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
    
>>> import platform
>>> platform.machine()
'x86'
>>> platform.version()
'5.1.2600'
>>> platform.platform()
'Windows-XP-5.1.2600-SP2'
>>> platform.uname()
('Windows', 'name', 'XP', '5.1.2600', 'x86', 'x86 Family 6 Model 15 Stepping 6, GenuineIntel')
>>> platform.system()
'Windows'
>>> platform.processor()
'x86 Family 6 Model 15 Stepping 6, GenuineIntel'
"""


def create_vocab(s_lab_df=None, data_dir: str = dataset_dir,
                 labelled_source_name: str = cfg['data']['train'],
                 unlabelled_source_name: str = cfg["data"]["source"]['unlabelled'],
                 # labelled_target_name=cfg['data']['test'],
                 unlabelled_target_name: str = cfg["data"]["target"]['unlabelled']):
    """ creates vocab and other info.

    :param data_dir:
    :param labelled_source_name:
    :param unlabelled_source_name:
    :param unlabelled_target_name:
    :return:
    """
    logger.info('creates vocab and other info.')
    ## Read source data
    if s_lab_df is None:
        # s_lab_df = read_labelled_json(data_dir, labelled_source_name)
        s_lab_df = read_csv(data_dir, labelled_source_name)
        s_lab_df = s_lab_df.sample(frac=1)

        # if labelled_source_name.startswith('fire16'):
        #     ## Match label space between two datasets:
        #     s_lab_df = labels_mapper(s_lab_df)

    token2label_vec_map = freq_tokens_per_class(s_lab_df)
    # label_vec = token_dist2token_labels(cls_freq, vocab_set)

    # s_unlab_df = json_keys2df(['text'], json_filename=unlabelled_source_name,
    #                           dataset_dir=data_dir)
    s_unlab_df = read_csv(data_file=unlabelled_source_name, data_dir=data_dir)

    # s_lab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_lab_df['domain'] = 0
    s_lab_df['labelled'] = True

    # s_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_unlab_df['domain'] = 0
    s_unlab_df['labelled'] = False

    ## Prepare source data
    s_unlab_df = s_unlab_df.append(s_lab_df[['text', 'domain', 'labelled']])

    S_dataname = unlabelled_source_name + "_data.csv"
    s_unlab_df.to_csv(join(data_dir, S_dataname))

    S_dataset, (S_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=S_dataname)

    S_vocab = {
        'freqs':       S_fields.vocab.freqs,
        'str2idx_map': dict(S_fields.vocab.stoi),
        'idx2str_map': S_fields.vocab.itos,
    }

    # logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
    logger.info("Source vocab size: [{}]".format(len(S_fields.vocab)))

    ## Read target data
    t_unlab_df = read_csv(data_dir, unlabelled_target_name)

    ## Prepare target data
    t_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    t_unlab_df['domain'] = 1
    t_unlab_df['labelled'] = False

    ## Target dataset
    T_dataname = unlabelled_target_name + "_data.csv"
    t_unlab_df.to_csv(join(data_dir, T_dataname))

    T_dataset, (T_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=T_dataname)
    logger.info("Target vocab size: [{}]".format(len(T_fields.vocab)))

    T_vocab = {
        'freqs':       T_fields.vocab.freqs,
        'str2idx_map': dict(T_fields.vocab.stoi),
        'idx2str_map': T_fields.vocab.itos,
    }

    ## Create combined data:
    c_df = s_unlab_df.append(t_unlab_df)

    c_dataname = unlabelled_source_name + '_' + unlabelled_target_name + "_data.csv"
    c_df.to_csv(join(data_dir, c_dataname))

    s_unlab_df = None
    t_unlab_df = None
    c_df = None

    C_dataset, (C_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=c_dataname)

    C_vocab = {
        'freqs':       C_fields.vocab.freqs,
        'str2idx_map': dict(C_fields.vocab.stoi),
        'idx2str_map': C_fields.vocab.itos,
    }

    ## Combine S and T vocabs:
    # C_vocab = get_c_vocab(S_vocab, T_vocab)
    # S_iter, T_iter = dataset2iter((S_dataset, T_dataset), batch_size=1)
    # c_iter = MultiIterator([S_iter, T_iter])
    logger.info("Combined vocab size: [{}]".format(len(C_vocab['str2idx_map'])))

    return C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
           T_dataset, T_fields, token2label_vec_map, s_lab_df


def main(data_dir: str = dataset_dir, lr=cfg["model"]["optimizer"]["lr"],
         mittens_iter: int = 300, gcn_hops: int = 5, glove_embs=None,
         labelled_source_name: str = cfg['data']['train'],
         labelled_val_name: str = cfg['data']['val'],
         unlabelled_source_name: str = cfg["data"]["source"]['unlabelled'],
         labelled_target_name: str = cfg['data']['test'],
         unlabelled_target_name: str = cfg["data"]["target"]['unlabelled'],
         train_batch_size=cfg['training']['train_batch_size'],
         test_batch_size=cfg['training']['eval_batch_size'], ):
    logger.critical(f'Current Learning Rate: [{lr}]')
    labelled_source_path = join(data_dir, labelled_source_name)
    unlabelled_source_name = unlabelled_source_name
    unlabelled_target_name = unlabelled_target_name
    S_dataname = unlabelled_source_name + "_data.csv"
    T_dataname = unlabelled_target_name + "_data.csv"

    if exists(labelled_source_path + 'S_vocab.json')\
            and exists(labelled_source_path + 'T_vocab.json')\
            and exists(labelled_source_path + 'labelled_token2vec_map.json'):
        # ## Read labelled source data
        # s_lab_df = read_labelled_json(data_dir, labelled_source_name)
        # ## Match label space between two datasets:
        # if str(labelled_source_name).startswith('fire16'):
        #     s_lab_df = labels_mapper(s_lab_df)

        C_vocab = read_json(labelled_source_path + 'C_vocab')
        S_vocab = read_json(labelled_source_path + 'S_vocab')
        T_vocab = read_json(labelled_source_path + 'T_vocab')
        labelled_token2vec_map = read_json(labelled_source_path + 'labelled_token2vec_map')

        if not exists(labelled_source_path + 'high_oov_freqs.json'):
            S_dataset, (S_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=S_dataname)
            T_dataset, (T_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=T_dataname)
    else:
        C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
        T_dataset, T_fields, labelled_token2vec_map, s_lab_df =\
            create_vocab(s_lab_df=None, data_dir=data_dir,
                         labelled_source_name=labelled_source_name,
                         unlabelled_source_name=unlabelled_source_name,
                         unlabelled_target_name=unlabelled_target_name)
        ## Save vocabs:
        save_json(C_vocab, labelled_source_path + 'C_vocab')
        save_json(S_vocab, labelled_source_path + 'S_vocab')
        save_json(T_vocab, labelled_source_path + 'T_vocab')
        save_json(labelled_token2vec_map, labelled_source_path + 'labelled_token2vec_map')

    if glove_embs is None:
        glove_embs = glove2dict()
    if exists(labelled_source_path + 'high_oov_freqs.json')\
            and exists(labelled_source_path + 'corpus.json')\
            and exists(labelled_source_path + 'corpus_toks.json'):
        high_oov_freqs = read_json(labelled_source_path + 'high_oov_freqs')
        # low_glove_freqs = read_json(labelled_source_name+'low_glove_freqs')
        corpus = read_json(labelled_source_path + 'corpus', convert_ordereddict=False)
        corpus_toks = read_json(labelled_source_path + 'corpus_toks', convert_ordereddict=False)
    else:
        ## Get all OOVs which does not have Glove embedding:
        high_oov_freqs, low_glove_freqs, corpus, corpus_toks =\
            preprocess_and_find_oov(
                (S_dataset, T_dataset), C_vocab, glove_embs=glove_embs,
                labelled_vocab_set=set(labelled_token2vec_map.keys()))

        ## Save token sets: high_oov_freqs, low_glove_freqs, corpus, corpus_toks
        save_json(high_oov_freqs, labelled_source_path + 'high_oov_freqs')
        # save_json(low_glove_freqs, labelled_source_name+'low_glove_freqs', overwrite=True)
        save_json(corpus, labelled_source_path + 'corpus')
        save_json(corpus_toks, labelled_source_path + 'corpus_toks')
        save_json(C_vocab, labelled_source_path + 'C_vocab', overwrite=True)

    ## Read labelled datasets and prepare:
    logger.info('Read labelled datasets and prepare')
    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab\
        = prepare_splitted_datasets()

    logger.info('Creating instance graphs')
    train_instance_graphs = Instance_Dataset_DGL(
        train_dataset, train_vocab, labelled_source_name, class_names=cfg[
            'data']['class_names'])
    logger.debug(train_instance_graphs.num_labels)
    # logger.debug(train_instance_graphs.graphs, train_instance_graphs.labels)

    train_dataloader = DataLoader(train_instance_graphs, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=train_instance_graphs.batch_graphs)

    logger.info(f"Number of training instance graphs: {len(train_instance_graphs)}")

    val_instance_graphs = Instance_Dataset_DGL(
        val_dataset, train_vocab, labelled_val_name, class_names=cfg[
            'data']['class_names'])

    val_dataloader = DataLoader(val_instance_graphs, batch_size=test_batch_size,
                                shuffle=True, collate_fn=val_instance_graphs.batch_graphs)

    logger.info(f"Number of validating instance graphs: {len(val_instance_graphs)}")

    test_instance_graphs = Instance_Dataset_DGL(
        test_dataset, train_vocab, labelled_target_name, class_names=cfg[
            'data']['class_names'])

    test_dataloader = DataLoader(test_instance_graphs, batch_size=test_batch_size, shuffle=True,
                                 collate_fn=test_instance_graphs.batch_graphs)

    logger.info(f"Number of testing instance graphs: {len(test_instance_graphs)}")

    ## Create token graph:
    logger.info(f'Creating token graph')
    g_ob = Token_Dataset_nx(corpus_toks, C_vocab, S_vocab, T_vocab, dataset_name=labelled_source_name)
    G = g_ob.G
    num_tokens = g_ob.num_tokens

    node_list = list(G.nodes)
    logger.info(f"Number of nodes {len(node_list)} and edges {len(G.edges)} in token graph")

    ## Create new embeddings for OOV tokens:
    oov_emb_filename = labelled_source_name + '_OOV_vectors_dict'
    if exists(join(data_dir, oov_emb_filename + '.pkl')):
        logger.info('Read OOV embeddings:')
        oov_embs = load_pickle(filepath=data_dir, filename=oov_emb_filename)
    else:
        logger.info('Create OOV embeddings using Mittens:')
        high_oov_tokens_list = list(high_oov_freqs.keys())
        c_corpus = corpus[0] + corpus[1]
        oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, c_corpus)
        oov_embs = train_mittens(oov_mat_coo, high_oov_tokens_list, glove_embs, max_iter=mittens_iter)
        save_pickle(oov_embs, filepath=data_dir, filename=oov_emb_filename, overwrite=True)

    ## Get adjacency matrix and node embeddings in same order:
    logger.info('Accessing token adjacency matrix')
    ## Note: Saving sparse tensor usually gets corrupted.
    # adj_filename = join(data_dir, labelled_source_name + "_adj.pt")
    # if exists(adj_filename):
    #     adj = load(adj_filename)
    #     # adj = sp_coo2torch_coo(adj)
    # else:
    #     adj = adjacency_matrix(G, nodelist=node_list, weight='weight')
    #     adj = sp_coo2torch_coo(adj)
    #     save(adj, adj_filename)
    adj = adjacency_matrix(G, nodelist=node_list, weight='weight')
    adj = sp_coo2torch_coo(adj)

    logger.info('Accessing token graph node embeddings:')
    emb_filename = join(data_dir, labelled_source_name + "_emb.pt")
    if exists(emb_filename):
        X = load(emb_filename)
    else:
        logger.info('Get node embeddings from token graph:')
        X = g_ob.get_node_embeddings(oov_embs, glove_embs, C_vocab['idx2str_map'])
        # X = sp_coo2torch_coo(X)
        save(X, emb_filename)

    # logger.info('Applying GCN Forward old')
    # X_hat = GCN_forward_old(adj, X, forward=gcn_hops)
    # logger.info('Applying GCN Forward')
    # X_hat = GCN_forward(adj, X, forward=gcn_hops)

    ## Apply Label Propagation to get label vectors for unlabelled nodes:
    logger.info('Getting propagated label vectors:')
    label_proba_filename = labelled_source_name + "_lpa_vecs.pt"
    if exists(label_proba_filename):
        lpa_vecs = torch.load(label_proba_filename)
    else:
        all_node_labels, labelled_masks = fetch_all_nodes(
            node_list, labelled_token2vec_map, C_vocab['idx2str_map'],
            default_fill=[0., 0., 0., 0.])
        lpa_vecs = label_propagation(adj, all_node_labels, labelled_masks)
        torch.save(lpa_vecs, label_proba_filename)

    logger.info('Recalculate edge weights using LPA vectors:')
    g_ob.normalize_edge_weights(lpa_vecs.numpy())

    adj = adjacency_matrix(g_ob.G, nodelist=node_list, weight='weight')
    adj = sp_coo2torch_coo(adj)

    ## Normalize Adjacency matrix:
    logger.info('Normalize token graph:')
    adj = g_ob.normalize_adj(adj)

    # ## Create label to propagated vector map:
    # logger.info('Create label to propagated vector map')
    # node_txt2label_vec = {}
    # for node_id in node_list:
    #     node_txt2label_vec[C_vocab['idx2str_map'][node_id]] =\
    #         lpa_vecs[node_id].tolist()
    # DataFrame.from_dict(node_txt2label_vec, orient='index').to_csv(labelled_source_name + 'node_txt2label_vec.csv')

    logger.info('Classifying combined graphs')
    train_epochs_output_dict, test_output = graph_multilabel_classification(
        adj, X, train_dataloader, val_dataloader, test_dataloader, num_tokens=num_tokens,
        in_feats=cfg['embeddings']['emb_dim'], hid_feats=cfg['gnn_params']['hid_dim'],
        num_heads=cfg['gnn_params']['num_heads'], epochs=cfg['training']['num_epoch'], lr=lr)

    # ## Propagating label vectors using GCN forward instead of LPA:
    # X_labels_hat = GCN_forward(adj, all_node_labels, forward=gcn_hops)
    # torch.save(X_labels_hat, 'X_labels_hat_05.pt')

    return C_vocab['str2idx_map']  # , X_hat


def prepare_datasets(train_df=None, test_df=None, stoi=None, vectors=None,
                     dim=cfg['embeddings']['emb_dim'], split_test=False,
                     get_iter=False, data_dir=dataset_dir,
                     train_filename=cfg['data']['train'],
                     test_filename=cfg['data']['test']):
    """ Creates train and test dataset from df and returns data loader.

    :param get_iter: If iterator over the text samples should be returned
    :param split_test: Splits the testing data
    :param train_df: Training dataframe
    :param test_df: Testing dataframe
    :param vectors: Custom Vectors for each token
    :param dim: Embedding dim
    :param data_dir:
    :param train_filename:
    :param test_filename:
    :return:
    """
    logger.info(f'Prepare labelled train (source) data: {train_filename}')
    if train_df is None:
        if train_filename.startswith('fire16'):
            train_df = load_fire16()
        else:
            train_df = read_labelled_json(data_dir, train_filename)

    train_dataname = train_filename + "_4class.csv"
    train_df.to_csv(join(data_dir, train_dataname))

    if stoi is None:
        logger.critical('Setting GLOVE vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1, labelled_data=True)
    else:
        logger.critical('Setting custom vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        train_vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    ## Plot representations:
    # plot_features_tsne(train_vocab.vocab.vectors,
    #                    list(train_vocab.vocab.stoi.keys()))

    # train_vocab = {
    #     'freqs':       train_vocab.vocab.freqs,
    #     'str2idx_map': dict(train_vocab.vocab.stoi),
    #     'idx2str_map': train_vocab.vocab.itos,
    #     'vectors': train_vocab.vocab.vectors,
    # }

    ## Prepare labelled target data:
    logger.info(f'Prepare labelled test (target) data: {test_filename}')
    if test_df is None:
        if test_filename.startswith('smerp17'):
            test_df = load_smerp17()
        else:
            test_df = read_labelled_json(data_dir, test_filename, data_set='test')

        if split_test:
            test_extra_df, test_df = split_target(df=test_df, test_size=0.4)
    test_dataname = test_filename + "_4class.csv"
    test_df.to_csv(join(data_dir, test_dataname))
    test_dataset, (test_vocab, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_dataname, labelled_data=True)

    # test_vocab = {
    #     'freqs':       test_vocab.vocab.freqs,
    #     'str2idx_map': dict(test_vocab.vocab.stoi),
    #     'idx2str_map': test_vocab.vocab.itos,
    #     'vectors': test_vocab.vocab.vectors,
    # }

    logger.info('Get iterator')
    if get_iter:
        train_batch_size = cfg['training']['train_batch_size']
        test_batch_size = cfg['training']['eval_batch_size']
        train_iter, val_iter = dataset2bucket_iter(
            (train_dataset, test_dataset), batch_sizes=(train_batch_size, test_batch_size))

        return train_dataset, test_dataset, train_vocab, test_vocab, train_iter, val_iter

    return train_dataset, test_dataset, train_vocab, test_vocab


def prepare_splitted_datasets(stoi=None, vectors=None, split_test=False, get_iter=False,
                              dim=cfg['embeddings']['emb_dim'],
                              data_dir=dataset_dir,
                              train_dataname=cfg["data"]["train"],
                              val_dataname=cfg["data"]["val"],
                              test_dataname=cfg["data"]["test"]):
    """ Creates train and test dataset from df and returns data loader.

    :param val_dataname:
    :param get_iter: If iterator over the text samples should be returned
    :param split_test: Splits the testing data
    :param train_df: Training dataframe
    :param test_df: Testing dataframe
    :param vectors: Custom Vectors for each token
    :param dim: Embedding dim
    :param data_dir:
    :param train_dataname:
    :param test_dataname:
    :return:
    """
    logger.info(f'Prepare labelled TRAINING (source) data: {train_dataname}')
    train_df, val_df, test_df = load_csvs(data_dir=data_dir, filenames=(
        train_dataname, val_dataname, test_dataname))
    logger.info(f"Train {train_df.shape}, Val {test_df.shape}, Test {val_df.shape}.")
    train_dataname = train_dataname + "_tmp.csv"
    train_df.to_csv(join(data_dir, train_dataname))

    if stoi is None:
        logger.critical('Setting default GLOVE vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1, labelled_data=True)
    else:
        logger.critical('Setting custom vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        train_vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    ## Prepare labelled validation data:
    logger.info(f'Prepare labelled VALIDATION (source) data: {val_dataname}')
    if val_df is None:
        val_df = read_csv(data_file=val_dataname, data_dir=data_dir)
        # val_df = read_labelled_json(data_dir, val_dataname, data_set='test')
    val_dataname = val_dataname + "_tmp.csv"
    val_df.to_csv(join(data_dir, val_dataname))
    val_dataset, (val_vocab, val_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=val_dataname, labelled_data=True)

    ## Prepare labelled target data:
    logger.info(f'Prepare labelled TESTING (target) data: {test_dataname}')
    if test_df is None:
        test_df = read_csv(data_file=test_dataname, data_dir=data_dir)
        # test_df = read_labelled_json(data_dir, test_dataname, data_set='test')

        # if split_test:
        #     test_extra_df, test_df = split_target(df=test_df, test_size=0.4)
    test_dataname = test_dataname + "_tmp.csv"
    test_df.to_csv(join(data_dir, test_dataname))
    test_dataset, (test_vocab, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_dataname, labelled_data=True)

    if get_iter:
        logger.info('Geting train, val and test iterators')
        train_batch_size = cfg['training']['train_batch_size']
        val_batch_size = cfg['training']['eval_batch_size']
        test_batch_size = cfg['training']['eval_batch_size']
        train_iter, val_iter, test_iter = dataset2bucket_iter(
            (train_dataset, val_dataset, test_dataset), batch_sizes=(
                train_batch_size, val_batch_size, test_batch_size))

        return train_dataset, val_dataset, test_dataset, train_vocab,\
               val_vocab, test_vocab, train_iter, val_iter, test_iter

    return train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab


def classify(train_df=None, test_df=None, stoi=None, vectors=None,
             n_classes=cfg['data']['num_classes'],
             dim=cfg['embeddings']['emb_dim'],
             data_dir=dataset_dir, train_filename=cfg['data']['train'],
             test_filename=cfg['data']['test'], cls_thresh=None,
             epoch=cfg['training']['num_epoch'], num_layers=cfg['lstm_params']['num_layers'],
             num_hidden_nodes=cfg['lstm_params']['hid_size'],
             dropout=cfg['model']['dropout'], default_thresh=0.5,
             lr=cfg['model']['optimizer']['lr'],
             train_batch_size=cfg['training']['train_batch_size'],
             test_batch_size=cfg['training']['eval_batch_size'],
             ):
    """

    :param n_classes:
    :param test_batch_size:
    :param train_df:
    :param test_df:
    :param stoi:
    :param vectors:
    :param dim:
    :param data_dir:
    :param train_filename:
    :param test_filename:
    :param cls_thresh:
    :param epoch:
    :param num_layers:
    :param num_hidden_nodes:
    :param dropout:
    :param default_thresh:
    :param lr:
    :param train_batch_size:
    :return:
    """
    ## Prepare labelled source data:
    # logger.info('Prepare labelled source data')
    # if train_df is None:
    #     train_df = read_labelled_json(data_dir, train_filename)
    #     train_df = labels_mapper(train_df)
    train_dataname = train_filename + "_4class.csv"
    train_df.to_csv(join(data_dir, train_dataname))

    if stoi is None:
        logger.critical('GLOVE features')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1, labelled_data=True)
    else:
        logger.critical('GCN features')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=1,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        train_vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    ## Plot representations:
    # plot_features_tsne(train_vocab.vocab.vectors,
    #                    list(train_vocab.vocab.stoi.keys()))

    ## Prepare labelled target data:
    logger.info('Prepare labelled target data')
    if test_df is None:
        test_df = read_labelled_json(data_dir, test_filename)
    test_dataname = test_filename + "_4class.csv"
    test_df.to_csv(join(data_dir, test_dataname))
    test_dataset, (test_vocab, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_dataname,  # init_vocab=True,
        labelled_data=True)

    # check whether cuda is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Get iterator')
    train_iter, val_iter = dataset2bucket_iter(
        (train_dataset, test_dataset), batch_sizes=(train_batch_size, test_batch_size))

    size_of_vocab = len(train_vocab.vocab)
    num_output_nodes = n_classes

    # instantiate the model
    logger.info('instantiate the model')
    model = BiLSTM_Classifier(size_of_vocab, num_hidden_nodes, num_output_nodes,
                              dim, num_layers, dropout=dropout)

    # architecture
    logger.info(model)

    # No. of trianable parameters
    logger.info('No. of trianable parameters')
    count_parameters(model)

    # Initialize the pretrained embedding
    logger.info('Initialize the pretrained embedding')
    pretrained_embeddings = train_vocab.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    logger.debug(pretrained_embeddings.shape)

    # label_cols = [str(cls) for cls in range(n_classes)]

    logger.info('Training model')
    model_best, val_preds_trues_best, val_preds_trues_all, losses = trainer(
        model, train_iter, val_iter, N_EPOCHS=epoch, lr=lr)

    plot_training_loss(losses['train'], losses['val'],
                       plot_name='loss' + str(epoch) + str(lr))

    if cls_thresh is None:
        cls_thresh = [default_thresh] * n_classes

    predicted_labels = logit2label(
        DataFrame(val_preds_trues_best['preds'].cpu().numpy()), cls_thresh,
        drop_irrelevant=False)

    logger.info('Calculate performance')
    result = calculate_performance_pl(val_preds_trues_best['trues'],
                                      val_preds_trues_best['preds'])

    logger.info("Result: {}".format(result))

    # result_df = flatten_results(result)
    # result_df.round(decimals=4).to_csv(
    #     join(data_dir, test_filename + '_results.csv'))

    return result


def get_supervised_result(model, train_iterator, val_iterator, test_iterator,
                          EPOCHS=5, cls_thresh=None, n_classes=cfg['data']['num_classes']):
    """ Train and Predict on full supervised mode.

    Returns:

    """

    model_best, val_preds_trues_best, val_preds_trues_all, losses = trainer(
        model, train_iterator, val_iterator, N_EPOCHS=EPOCHS)

    # logger.debug(losses)

    # evaluate the model
    test_loss, test_preds_trues = predict_with_label(model_best, test_iterator)

    if cls_thresh is None:
        cls_thresh = [0.5] * n_classes

    predicted_labels = logit2label(
        DataFrame(test_preds_trues['preds'].numpy()), cls_thresh,
        drop_irrelevant=False)

    result = calculate_performance_pl(test_preds_trues['trues'],
                                      predicted_labels)

    logger.info("Supervised result: {}".format(dumps(result, indent=4)))
    return result, model_best


def save_glove(glove_embs, glove_dir=cfg["paths"]["embedding_dir"][plat][user],
               glove_file='oov_glove.txt'):
    """

    :param glove_embs:
    :param glove_dir:
    :param glove_file:
    """
    with open(join(glove_dir, glove_file), 'w', encoding='UTF-8') as glove_f:
        for token, vec in glove_embs.items():
            line = token + ' ' + str(vec)
            glove_f.write(line)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("-d", "--dataset_name",
    #                     default=cfg['data']['source']['labelled'], type=str)
    # parser.add_argument("-m", "--model_name",
    #                     default=cfg['model']['model_name'], type=str)
    # parser.add_argument("-mt", "--model_type",
    #                     default=cfg['model']['model_type'], type=str)
    # parser.add_argument("-ne", "--num_train_epochs",
    #                     default=cfg['training']['num_epoch'], type=int)
    # parser.add_argument("-c", "--use_cuda",
    #                     default=cfg['model']['use_cuda'], action='store_true')
    #
    # args = parser.parse_args()

    data_dir = dataset_dir

    # from File_Handlers.read_datasets import load_fire16, load_smerp17

    # from File_Handlers.read_datasets import load_fire16, load_smerp17
    #
    # if args.dataset_name.startswith('fire16'):
    #     train_df = load_fire16()
    #
    # if cfg['data']['target']['labelled'].startswith('smerp17'):
    #     target_df = load_smerp17()
    #     target_train_df, test_df = split_target(df=target_df, test_size=0.999)

    # result, model_outputs = BERT_classifier(
    #     train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
    #     model_name=args.model_name, model_type=args.model_type,
    #     num_epoch=args.num_train_epochs, use_cuda=True)

    # glove_embs = glove2dict()

    # train, test = split_target(test_df, test_size=0.3,
    #                            train_size=.6, stratified=False)
    lrs = [1e-3, 1e-4, 1e-5]
    for lr in lrs:
        s2i_dict = main(lr=lr)

    exit(0)

    glove_target = classify(
        train_df=train_df, test_df=test_df, epoch=2, num_layers=2,
        num_hidden_nodes=50, dropout=.3, lr=3e-4)
    logger.info(glove_target)
    gcn_target = classify(
        train_df=train_df, test_df=test_df, stoi=s2i_dict,
        vectors=X_hat, epoch=2, num_hidden_nodes=50,
        num_layers=2, dropout=.3, lr=3e-4)
    logger.info(gcn_target)

    epochs = [10, 25]
    layer_sizes = [1, 4]
    gcn_forward = [6, 2]
    hid_dims = [50]
    dropouts = [0.5]
    lrs = [1e-5, 1e-6]

    final_result = []

    train_portions = [0.2, 0.5, 1.0]
    for train_portion in train_portions:
        train, test = split_target(test_df, test_size=0.3,
                                   train_size=train_portion, stratified=False)
        for c in gcn_forward:
            s2i_dict, X_hat = main(gcn_hops=c, glove_embs=glove_embs)

            for a in epochs:
                for b in layer_sizes:
                    for d in hid_dims:
                        for e in dropouts:
                            for f in lrs:
                                logger.critical(f'Epoch: [{a}], LSTM #layers: '
                                                f'[{b}], GCN forward: [{c}], '
                                                f'Hidden'
                                                f' dims: [{d}], Dropouts: ['
                                                f'{e}], '
                                                f'Learning Rate: [{f}], ')
                                params = {
                                    'Epoch':         a,
                                    'LSTM #layers':  b,
                                    'GCN forward':   c,
                                    'Hidden dims':   d,
                                    'Dropouts':      e,
                                    'Learning Rate': f,
                                    'train_portion': train_portion,
                                }
                                glove_target = classify(
                                    train_df=train, test_df=test, epoch=a,
                                    num_layers=b, num_hidden_nodes=d, dropout=e,
                                    lr=f)

                                glove_source = classify(
                                    test_df=test, epoch=a, num_hidden_nodes=d,
                                    num_layers=b, dropout=e, lr=f, )

                                gcn_target = classify(
                                    train_df=train, test_df=test, stoi=s2i_dict,
                                    vectors=X_hat, epoch=a, num_hidden_nodes=d,
                                    num_layers=b, dropout=e, lr=f, )

                                gcn_source = classify(
                                    test_df=test, stoi=s2i_dict, vectors=X_hat,
                                    epoch=a, num_hidden_nodes=d, num_layers=b,
                                    dropout=e, lr=f, )

                                result_dict = {
                                    'params':       params,
                                    'glove_target': glove_target,
                                    'glove_source': glove_source,
                                    'gcn_target':   gcn_target,
                                    'gcn_source':   gcn_source,
                                }
                                final_result.append(result_dict)

                                logger.info("Result: {}".format(result_dict))

    logger.info("ALL Results: {}".format(final_result))

    # present_result(final_result)

    # main(mittens_iter=200, gcn_hops=1, epoch=20, num_layers=1,
    #      num_hidden_nodes=100, dropout=0.2, lr=1e-4)
    # main(mittens_iter=200, gcn_hops=1, epoch=20, num_layers=1,
    #      num_hidden_nodes=100, dropout=0.2, lr=1e-5)
    #
    # main(mittens_iter=500, gcn_hops=1, epoch=20, num_layers=2,
    #      num_hidden_nodes=100, dropout=0.3, lr=1e-4)
    # main(mittens_iter=500, gcn_hops=3, epoch=20, num_layers=3,
    #      num_hidden_nodes=120, dropout=0.4, lr=1e-5)
    # main(mittens_iter=500, gcn_hops=5, epoch=10, num_layers=3,
    #      num_hidden_nodes=50, dropout=0.5, lr=1e-6)
    #
    # main(mittens_iter=1000, gcn_hops=3, epoch=10, num_layers=1,
    #      num_hidden_nodes=100, dropout=0.2, lr=1e-4)
    # main(mittens_iter=1000, gcn_hops=3, epoch=10, num_layers=2,
    #      num_hidden_nodes=100, dropout=0.2, lr=1e-5)

    logger.info("Execution complete.")
