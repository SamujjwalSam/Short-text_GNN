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
import argparse
import pandas as pd
import networkx as nx
from pathlib import Path
from os import environ
from os.path import join, exists
from collections import OrderedDict
from json import dumps
from scipy import sparse

# from Label_Propagation_PyTorch.label_propagation import fetch_all_nodes, label_propagation
from Utils.utils import count_parameters, logit2label, split_df,\
    freq_tokens_per_class, merge_dicts, split_target
from Layers.BiLSTM_Classifier import BiLSTM_Classifier
from File_Handlers.csv_handler import read_tweet_csv
from File_Handlers.json_handler import save_json, read_json, json_keys2df,\
    read_labelled_json
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Data_Handlers.torchtext_handler import dataset2bucket_iter
from Data_Handlers.graph_constructor_dgl import DGL_Graph
from build_corpus_vocab import get_dataset_fields
from Data_Handlers.graph_constructor_nx import get_node_features,\
    add_edge_weights, create_tokengraph, generate_sample_subgraphs
from Layers.GCN_forward import GCN_forward, GCN_forward_old
from GNN_DGL.GNN_dgl import graph_multilabel_classification
from Transformers_simpletransformers.BERT_multilabel_classifier import BERT_classifier
from finetune_static_embeddings import glove2dict, calculate_cooccurrence_mat,\
    train_mittens, preprocess_and_find_oov
from Trainer.trainer import trainer, predict_with_label
from Class_mapper.FIRE16_SMERP17_map import labels_mapper
from Plotter.plot_functions import plot_training_loss, plot_graph
from Metrics.metrics import calculate_performance_pl
from config import configuration as cfg, platform as plat, username as user
from Logger.logger import logger

n_classes = 4

## Enable multi GPU cuda environment:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

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
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
    
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


def create_vocab(s_lab_df=None,
                 data_dir: str = cfg["paths"]["dataset_dir"][plat][user],
                 labelled_source_name: str = cfg["data"]["source"]['labelled'],
                 unlabelled_source_name: str = cfg["data"]["source"]['unlabelled'],
                 # labelled_target_name=cfg["data"]["target"]['labelled'],
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
        s_lab_df = read_labelled_json(data_dir, labelled_source_name)

        if labelled_source_name.startswith('fire16'):
            ## Match label space between two datasets:
            s_lab_df = labels_mapper(s_lab_df)

    token2label_vec_map = freq_tokens_per_class(s_lab_df)
    # label_vec = token_dist2token_labels(cls_freq, vocab_set)

    s_unlab_df = json_keys2df(['text'], json_filename=unlabelled_source_name,
                              dataset_dir=data_dir)

    # s_lab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_lab_df['domain'] = 0
    s_lab_df['labelled'] = True

    # s_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    s_unlab_df['domain'] = 0
    s_unlab_df['labelled'] = False

    ## Prepare source data
    s_unlab_df = s_unlab_df.append(s_lab_df[['text', 'domain', 'labelled']])

    S_data_name = unlabelled_source_name + "_data.csv"
    s_unlab_df.to_csv(join(data_dir, S_data_name))

    S_dataset, (S_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=S_data_name)

    S_vocab = {
        'freqs':       S_fields.vocab.freqs,
        'str2idx_map': dict(S_fields.vocab.stoi),
        'idx2str_map': S_fields.vocab.itos,
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

    T_dataset, (T_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=T_data_name)
    logger.info("Target vocab size: [{}]".format(len(T_fields.vocab)))

    T_vocab = {
        'freqs':       T_fields.vocab.freqs,
        'str2idx_map': dict(T_fields.vocab.stoi),
        'idx2str_map': T_fields.vocab.itos,
    }

    ## Create combined data:
    c_df = s_unlab_df.append(t_unlab_df)

    c_data_name = unlabelled_source_name + '_' + unlabelled_target_name\
                  + "_data.csv"
    c_df.to_csv(join(data_dir, c_data_name))

    s_unlab_df = None
    t_unlab_df = None
    c_df = None

    C_dataset, (C_fields, LABEL) = get_dataset_fields(csv_dir=data_dir,
                                                      csv_file=c_data_name)

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


def main(data_dir: str = cfg["paths"]["dataset_dir"][plat][user],
         labelled_source_name: str = cfg["data"]["source"]['labelled'],
         unlabelled_source_name: str = cfg["data"]["source"]['unlabelled'],
         labelled_target_name: str = cfg["data"]["target"]['labelled'],
         unlabelled_target_name: str = cfg["data"]["target"]['unlabelled'],
         mittens_iter: int = 1000, gcn_hops: int = 5,
         glove_embs=None):
    if glove_embs is None:
        glove_embs = glove2dict()
    data_dir = Path(data_dir)
    labelled_source_name = labelled_source_name
    unlabelled_source_name = unlabelled_source_name
    unlabelled_target_name = unlabelled_target_name
    S_data_name = Path(unlabelled_source_name + "_data.csv")
    T_data_name = Path(unlabelled_target_name + "_data.csv")

    if exists(labelled_source_name + 'S_vocab.json') and exists(labelled_source_name + 'T_vocab.json') and exists(
            labelled_source_name + 'labelled_token2vec_map.json'):
        # ## Read labelled source data
        # s_lab_df = read_labelled_json(data_dir, labelled_source_name)
        # ## Match label space between two datasets:
        # if str(labelled_source_name).startswith('fire16'):
        #     s_lab_df = labels_mapper(s_lab_df)

        C_vocab = read_json(labelled_source_name + 'C_vocab')
        S_vocab = read_json(labelled_source_name + 'S_vocab')
        T_vocab = read_json(labelled_source_name + 'T_vocab')
        labelled_token2vec_map = read_json(labelled_source_name + 'labelled_token2vec_map')

        if not exists(labelled_source_name + 'high_oov_freqs.json'):
            S_dataset, (S_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=S_data_name)
            T_dataset, (T_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=T_data_name)
    else:
        C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
        T_dataset, T_fields, labelled_token2vec_map, s_lab_df =\
            create_vocab(s_lab_df=None, data_dir=data_dir,
                         labelled_source_name=labelled_source_name,
                         unlabelled_source_name=unlabelled_source_name,
                         unlabelled_target_name=unlabelled_target_name)
        ## Save vocabs:
        save_json(C_vocab, labelled_source_name + 'C_vocab', overwrite=True)
        save_json(S_vocab, labelled_source_name + 'S_vocab', overwrite=True)
        save_json(T_vocab, labelled_source_name + 'T_vocab', overwrite=True)
        save_json(labelled_token2vec_map, labelled_source_name + 'labelled_token2vec_map',
                  overwrite=True)

    if exists(labelled_source_name + 'high_oov_freqs.json') and exists(labelled_source_name + 'corpus.json') and\
            exists(labelled_source_name + 'corpus_toks.json'):
        high_oov_freqs = read_json(labelled_source_name + 'high_oov_freqs')
        # low_glove_freqs = read_json(labelled_source_name+'low_glove_freqs')
        corpus = read_json(labelled_source_name + 'corpus', convert_ordereddict=False)
        corpus_toks = read_json(labelled_source_name + 'corpus_toks', convert_ordereddict=False)
    else:
        ## Get all OOVs which does not have Glove embedding:
        high_oov_freqs, low_glove_freqs, corpus, corpus_toks =\
            preprocess_and_find_oov(
                (S_dataset, T_dataset), C_vocab, glove_embs=glove_embs,
                labelled_vocab_set=set(labelled_token2vec_map.keys()))

        ## Save token sets: high_oov_freqs, low_glove_freqs, corpus, corpus_toks
        save_json(high_oov_freqs, labelled_source_name + 'high_oov_freqs', overwrite=True)
        # save_json(low_glove_freqs, labelled_source_name+'low_glove_freqs', overwrite=True)
        save_json(corpus, labelled_source_name + 'corpus', overwrite=True)
        save_json(corpus_toks, labelled_source_name + 'corpus_toks', overwrite=True)

        save_json(C_vocab, labelled_source_name + 'C_vocab', overwrite=True)

    graph_path = labelled_source_name + "G.pkl"
    if exists(graph_path):
        G = nx.read_gpickle(graph_path)
    else:
        G = create_tokengraph(corpus_toks, C_vocab, S_vocab, T_vocab)
        ## Calculate edge weights from cooccurrence stats:
        G = add_edge_weights(G)
        nx.write_gpickle(G, graph_path)

    logger.info("Number of nodes: [{}]".format(len(G.nodes)))
    logger.info("Number of edges: [{}]".format(len(G.edges)))
    node_list = list(G.nodes)

    ## Create new embeddings for OOV tokens:
    oov_filename = labelled_source_name + '_OOV_vectors_dict'
    if exists(join(data_dir, oov_filename + '.pkl')):
        logger.info('Read embeddings for OOV tokens')
        oov_embs = load_pickle(pkl_file_path=data_dir,
                               pkl_file_name=oov_filename)
    else:
        logger.info('Create new embeddings for OOV tokens')
        high_oov_tokens_list = list(high_oov_freqs.keys())
        c_corpus = corpus[0] + corpus[1]
        oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, c_corpus)
        oov_embs = train_mittens(oov_mat_coo, high_oov_tokens_list, glove_embs, max_iter=mittens_iter)
        save_pickle(oov_embs, filepath=data_dir, filename=oov_filename, overwrite=True)

    logger.info('Creating instance graphs')
    from Data_Handlers.graph_data_handler_dgl import Graph_Data_Handler

    gdh = Graph_Data_Handler(dataset_dir=data_dir, dataset_info=cfg['data'])
    gdh.setup(stage='da')
    # gdh.train_dataloader()

    logger.info('Classifying instance graphs')
    train_epochs_output_dict, test_output = graph_multilabel_classification(
        gdh, in_feats=100, hid_feats=cfg['gnn_params']['hid_dim'],
        num_heads=cfg['gnn_params']['num_heads'], epochs=cfg['training']['num_epoch'])

    ## TODO: Generate <UNK> embedding from low freq tokens:

    ## Get adjacency matrix and node embeddings in same order:
    logger.info('Accessing Adjacency matrix')
    adj_filename = labelled_source_name + "adj.npz"
    if exists(adj_filename):
        adj = sparse.load_npz(adj_filename)
    else:
        adj = nx.adjacency_matrix(G, nodelist=node_list, weight='edge_weight')
        # adj_np = nx.to_numpy_matrix(G)
        sparse.save_npz(adj_filename, adj)

    # ## Apply Label Propagation to get label vectors for unlabelled nodes:
    # logger.info('Apply Label Propagation to get label vectors for unlabelled nodes')
    # all_node_labels, labelled_masks = fetch_all_nodes(
    #     node_list, labelled_token2vec_map, C_vocab['idx2str_map'],
    #     default_fill=[0., 0., 0., 0.])
    #
    # label_proba_filename = labelled_source_name + "labels_propagated.pt"
    # if exists(label_proba_filename):
    #     labels_propagated = torch.load(label_proba_filename)
    # else:
    #     labels_propagated = label_propagation(adj, all_node_labels,
    #                                           labelled_masks)
    #     torch.save(labels_propagated, label_proba_filename)
    #
    # # Create label to propagated vector map:
    # logger.info('Create label to propagated vector map')
    # node_txt2label_vec = {}
    # for node_id in node_list:
    #     node_txt2label_vec[C_vocab['idx2str_map'][node_id]] =\
    #         labels_propagated[node_id].tolist()
    # pd.DataFrame.from_dict(node_txt2label_vec, orient='index').to_csv(labelled_source_name + 'node_txt2label_vec.csv')

    # ## Propagating label vectors using GCN forward instead of LPA:
    # X_labels_hat = GCN_forward(adj, all_node_labels, forward=gcn_hops)
    # torch.save(X_labels_hat, 'X_labels_hat_05.pt')

    logger.info('Get node features')
    merged_embs = merge_dicts(glove_embs, oov_embs)
    # save_pickle(merged_embs, cfg["embeddings"]["saved_emb_file"], data_dir)
    X = get_node_features(merged_embs, oov_embs, C_vocab['idx2str_map'], node_list)
    logger.info('Applying GCN Forward old')
    X_hat = GCN_forward_old(adj, X, forward=gcn_hops)
    # logger.info('Applying GCN Forward')
    # X_hat = GCN_forward(adj, X, forward=gcn_hops)

    return C_vocab['str2idx_map'], X_hat


def classify(train_df=None, test_df=None, stoi=None, vectors=None,
             dim=cfg['prep_vecs']['input_size'],
             data_dir=cfg["paths"]["dataset_dir"][plat][user],
             train_filename=cfg["data"]["source"]['labelled'],
             test_filename=cfg["data"]["target"]['labelled'],
             cls_thresh=None, epoch=cfg['training']['num_epoch'],
             num_layers=cfg['lstm_params']['num_layers'],
             num_hidden_nodes=cfg['lstm_params']['hid_size'],
             dropout=cfg['model']['dropout'], default_thresh=0.5,
             lr=cfg['model']['optimizer']['lr'],
             train_batch_size=cfg['training']['train_batch_size'],
             test_batch_size=cfg['training']['eval_batch_size'],
             ):
    """

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
    train_data_name = train_filename + "_4class_data.csv"
    train_df.to_csv(join(data_dir, train_data_name))

    if stoi is None:
        logger.critical('GLOVE features')
        train_dataset, (train_fields, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_data_name, min_freq=1, labelled_data=True)
    else:
        logger.critical('GCN features')
        train_dataset, (train_fields, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_data_name, min_freq=1,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        train_fields.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    ## Plot representations:
    # plot_features_tsne(train_fields.vocab.vectors,
    #                    list(train_fields.vocab.stoi.keys()))

    ## Prepare labelled target data:
    logger.info('Prepare labelled target data')
    if test_df is None:
        test_df = read_labelled_json(data_dir, test_filename)
    test_data_name = test_filename + "_4class_data.csv"
    test_df.to_csv(join(data_dir, test_data_name))
    test_dataset, (test_fields, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_data_name,  # init_vocab=True,
        labelled_data=True)

    # check whether cuda is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Get iterator')
    train_iter, val_iter = dataset2bucket_iter(
        (train_dataset, test_dataset), batch_sizes=(train_batch_size, test_batch_size))

    size_of_vocab = len(train_fields.vocab)
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
    pretrained_embeddings = train_fields.vocab.vectors
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
        pd.DataFrame(val_preds_trues_best['preds'].cpu().numpy()), cls_thresh,
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
                          EPOCHS=5, cls_thresh=None):
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
        pd.DataFrame(test_preds_trues['preds'].numpy()), cls_thresh,
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
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-d", "--dataset_name",
                        default=cfg['data']['source']['labelled'], type=str)
    parser.add_argument("-m", "--model_name",
                        default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type",
                        default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-ne", "--num_train_epochs",
                        default=cfg['training']['num_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['model']['use_cuda'], action='store_true')

    args = parser.parse_args()

    data_dir = cfg["paths"]["dataset_dir"][plat][user]

    from File_Handlers.read_datasets import load_fire16, load_smerp17

    if args.dataset_name.startswith('fire16'):
        train_df = load_fire16()

    if cfg['data']['target']['labelled'].startswith('smerp17'):
        target_df = load_smerp17()
        target_train_df, test_df = split_target(df=target_df, test_size=0.4)

    result, model_outputs = BERT_classifier(
        train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
        model_name=args.model_name, model_type=args.model_type,
        num_epoch=args.num_train_epochs, use_cuda=True)

    exit(0)

    glove_embs = glove2dict()

    # train, test = split_target(test_df, test_size=0.3,
    #                            train_size=.6, stratified=False)
    s2i_dict, X_hat = main(gcn_hops=2, glove_embs=glove_embs)

    glove_target = classify(
        train_df=train_df, test_df=test_df, epoch=2, num_layers=2,
        num_hidden_nodes=50, dropout=.3, lr=3e-4)
    logger.info(glove_target)
    gcn_target = classify(
        train_df=train_df, test_df=test_df, stoi=s2i_dict,
        vectors=X_hat, epoch=2, num_hidden_nodes=50,
        num_layers=2, dropout=.3, lr=3e-4)
    logger.info(gcn_target)

    exit(0)

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
