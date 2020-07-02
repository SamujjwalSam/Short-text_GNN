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
import argparse
import pandas as pd
import networkx as nx
from os import environ
from os.path import join, exists
from collections import OrderedDict
from json import dumps

from Utils.utils import count_parameters, logit2label, calculate_performance,\
    split_df, token_class_proba
from Layers.BiLSTM_Classifier import BiLSTM_Classifier
from File_Handlers.csv_handler import read_tweet_csv
from File_Handlers.json_handler import json_keys2df, read_labelled_json
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Data_Handlers.torchtext_handler import dataset2bucket_iter
from build_corpus_vocab import get_dataset_fields
from Data_Handlers.graph_data_handler import get_node_features,\
    add_edge_weights, create_tokengraph, get_label_vectors
from Layers.GCN_forward import GCN_forward, netrowkx2geometric
from Layers.BERT_multilabel_classifier import BERT_classifier
from finetune_static_embeddings import glove2dict, calculate_cooccurrence_mat,\
    train_model, preprocess_and_find_oov
from Trainer.Training import trainer, predict_with_label
from Class_mapper.FIRE16_SMERP17_map import labels_mapper
from Plotter.plot_functions import plot_training_loss
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


def map_nodetxt2GCNvec(G, node_list, X, id2txt_map, return_format='pytorch'):
    """ Creates a dict from token (node_id) to it's vectors.

    :param G:
    :param node_list:
    :param X:
    :param return_format:
    :return:
    """
    GCNvec_dict = OrderedDict()
    # stoi_dict = OrderedDict()
    for i, node_id in enumerate(node_list):
        # node_txt = G.nodes[node_id]['node_txt']
        node_txt = id2txt_map[node_id]
        # stoi_dict[node_txt] = node_id
        if return_format == 'numpy':
            GCNvec_dict[node_txt] = X[i].numpy()
        elif return_format == 'list':
            GCNvec_dict[node_txt] = X[i].tolist()
        elif return_format == 'pytorch':
            GCNvec_dict[node_txt] = X[i]
        else:
            raise NotImplementedError(f'Unknown format: [{return_format}].')

    return GCNvec_dict  # , stoi_dict


def split_target(df=None,
                 data_dir=cfg["paths"]["dataset_dir"][plat][user],
                 labelled_data_name=cfg["data"]["target"]['labelled'],
                 test_size=0.3, train_size=1.0, n_classes=4):
    """ Splits labelled target data to train and test set.

    :param data_dir:
    :param labelled_data_name:
    :param test_size:
    :param train_size:
    :param n_classes:
    :return:
    """
    ## Read target data
    if df is None:
        df = read_labelled_json(data_dir, labelled_data_name)
    df, t_lab_test_df = split_df(df, test_size=test_size,
                                 stratified=True, order=2,
                                 n_classes=n_classes)

    logger.info(f'Number of TEST samples: [{t_lab_test_df.shape[0]}]')

    if train_size is not None:
        _, df = split_df(df, test_size=train_size,
                         stratified=True, order=2, n_classes=n_classes)
    logger.info(f'Number of TRAIN samples: [{df.shape[0]}]')

    # token_dist(t_lab_df)

    return df, t_lab_test_df


def main(data_dir=cfg["paths"]["dataset_dir"][plat][user],
         labelled_source_name=cfg["data"]["source"]['labelled'],
         unlabelled_source_name=cfg["data"]["source"]['unlabelled'],
         # labelled_target_name=cfg["data"]["target"]['labelled'],
         unlabelled_target_name=cfg["data"]["target"]['unlabelled'],
         mittens_iter=1000, gcn_hops=5,
         glove_embs=glove2dict()):
    ## Read source data
    s_lab_df = read_labelled_json(data_dir, labelled_source_name)

    s_lab_df = labels_mapper(s_lab_df)

    token2label_vec_map = token_class_proba(s_lab_df)
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
                                                      csv_file=S_data_name, )

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
                                                      csv_file=T_data_name,
                                                      )
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
                                                      csv_file=c_data_name, )

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

    ## Get all OOVs which does not have Glove embedding:
    high_oov_freqs, low_glove_freqs, corpus, corpus_toks =\
        preprocess_and_find_oov(
            (S_dataset, T_dataset), C_vocab, glove_embs=glove_embs,
            labelled_vocab_set=set(token2label_vec_map.keys()))

    # ## Create token graph G using source data:
    # G = create_src_tokengraph(corpus_toks[0], S_vocab)
    #
    # ## Add nodes and edges to the token graph generated using source data:
    # G, c_i2s = create_tgt_tokengraph(corpus_toks[1], T_vocab, S_vocab, G)
    G = create_tokengraph(corpus_toks, C_vocab, S_vocab, T_vocab)

    ## Calculate edge weights from cooccurrence stats:
    G = add_edge_weights(G)

    logger.info("Number of nodes in the token graph: [{}]".format(len(G.nodes)))
    logger.info("Number of edges in the token graph: [{}]".format(len(G.edges)))

    ## Create new embeddings for OOV tokens:
    oov_filename = labelled_source_name + '_OOV_vectors_dict'
    if exists(join(data_dir, oov_filename + '.pkl')):
        oov_embs = load_pickle(pkl_file_path=data_dir,
                               pkl_file_name=oov_filename)
    else:
        high_oov_tokens_list = list(high_oov_freqs.keys())
        c_corpus = corpus[0] + corpus[1]
        oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, c_corpus)
        oov_embs = train_model(oov_mat_coo, high_oov_tokens_list, glove_embs,
                               max_iter=mittens_iter)
        save_pickle(oov_embs, pkl_file_path=data_dir,
                    pkl_file_name=oov_filename, )

    node_list = G.nodes

    X_labels = get_label_vectors(node_list, token2label_vec_map, C_vocab[
        'idx2str_map'])
    torch.save(X_labels, 'X_labels_05.pt')

    # glove_embs = merge_dicts(glove_embs, oov_embs)

    ## TODO: Generate <UNK> embedding from low freq tokens:
    ## Save embedding with OOV tokens:
    # save_pickle(glove_embs, pkl_file_name=cfg["embeddings"][
    # "embedding_file"] +
    #                                       labelled_source_filename + '_' +
    #                                       labelled_target_filename,
    #             pkl_file_path=cfg["paths"]["embedding_dir"][plat][user])

    # save_glove(glove_embs, glove_dir=cfg["paths"]["embedding_dir"][plat][
    # user],
    #            glove_file='oov_glove.txt')

    ## Get adjacency matrix and node embeddings in same order:
    adj = nx.adjacency_matrix(G, nodelist=node_list, weight='weight')
    # adj_np = nx.to_numpy_matrix(G)
    X_labels_hat = GCN_forward(adj, X_labels, forward=gcn_hops)
    torch.save(X_labels_hat, 'X_labels_hat_05.pt')

    X = get_node_features(glove_embs, oov_embs, C_vocab['idx2str_map'],
                          node_list)
    X_hat = GCN_forward(adj, X, forward=gcn_hops)

    ## Create text to GCN forward vectors:
    # X_dict = map_nodetxt2GCNvec(G, node_list, X_hat, C_vocab['idx2str_map'])

    ## Save GCN forwarded vectors for future use:
    # save_pickle(X_dict, pkl_file_name='X_dict.t', pkl_file_path=data_dir)
    # torch.save(X_dict, join(data_dir, 'X_dict.pt'))
    # X_dict = torch.load(join(data_dir, 'X_dict.pt'))
    # save_glove(glove_embs, glove_dir=cfg["paths"]["embedding_dir"][plat][
    # user],
    #            glove_file='oov_glove.txt')

    ## Construct tweet subgraph:
    # S_iter, T_iter = dataset2iter((S_dataset, T_dataset), batch_size=1)
    # txts_subgraphs = generate_sample_subgraphs(s_lab_df.text.to_list(), G=G)
    # logger.info("Fetching subgraph: [{}]".format(txts_subgraphs))
    # print(txts_subgraphs[0].nodes)
    # plot_graph(txts_subgraphs[0])

    return C_vocab['str2idx_map'], X_hat


def classify(train_df=None, test_df=None, stoi=None, vectors=None,
             dim=cfg['prep_vecs']['input_size'],
             data_dir=cfg["paths"]["dataset_dir"][plat][user],
             train_filename=cfg["data"]["source"]['labelled'],
             test_filename=cfg["data"]["target"]['labelled'],
             cls_thresh=None, epoch=cfg['sampling']['num_epochs'],
             num_layers=cfg['lstm_params']['num_layers'],
             num_hidden_nodes=cfg['lstm_params']['hid_size'],
             dropout=cfg['model']['dropout'], default_thresh=0.5,
             lr=cfg['model']['optimizer']['learning_rate'],
             train_batch_size=128,
             ):
    ## Prepare labelled source data:
    if train_df is None:
        train_df = read_labelled_json(data_dir, train_filename)
        train_df = labels_mapper(train_df)
    train_data_name = train_filename + "_4class_data.csv"
    train_df.to_csv(join(data_dir, train_data_name))

    if stoi is None:
        logger.critical('simple GLOVE features')
        train_dataset, (train_fields, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_data_name, min_freq=1,
            labelled_data=True)
    else:
        logger.critical('GCN features')
        train_dataset, (train_fields, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_data_name, min_freq=1,
            labelled_data=True, embedding_file=None,
            embedding_dir=None)
        train_fields.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    ## Plot representations:
    # plot_features_tsne(train_fields.vocab.vectors,
    #                    list(train_fields.vocab.stoi.keys()))

    ## Prepare labelled target data:
    if test_df is None:
        test_df = read_labelled_json(data_dir, test_filename)
    test_data_name = test_filename + "_4class_data.csv"
    test_df.to_csv(join(data_dir, test_data_name))
    test_dataset, (test_fields, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_data_name,  # init_vocab=True,
        labelled_data=True)

    # check whether cuda is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter, val_iter = dataset2bucket_iter(
        (train_dataset, test_dataset), batch_sizes=(train_batch_size,
                                                    train_batch_size * 2))

    size_of_vocab = len(train_fields.vocab)
    num_output_nodes = n_classes

    # instantiate the model
    model = BiLSTM_Classifier(size_of_vocab, num_hidden_nodes, num_output_nodes,
                              dim, num_layers, dropout=dropout)

    # architecture
    logger.info(model)

    # No. of trianable parameters
    count_parameters(model)

    # Initialize the pretrained embedding
    pretrained_embeddings = train_fields.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    logger.debug(pretrained_embeddings.shape)

    # label_cols = [str(cls) for cls in range(n_classes)]

    model_best, val_preds_trues_best, val_preds_trues_all, losses = trainer(
        model, train_iter, val_iter, N_EPOCHS=epoch, lr=lr)

    plot_training_loss(losses['train'], losses['val'],
                       plot_name='loss' + str(epoch) + str(lr))

    if cls_thresh is None:
        cls_thresh = [default_thresh] * n_classes

    predicted_labels = logit2label(
        pd.DataFrame(val_preds_trues_best['preds'].cpu().numpy()), cls_thresh,
        drop_irrelevant=False)

    result = calculate_performance(val_preds_trues_best['trues'].cpu().numpy(),
                                   predicted_labels)

    logger.info("Result: {}".format(dumps(result, indent=4)))

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

    result = calculate_performance(test_preds_trues['trues'].numpy(),
                                   predicted_labels)

    logger.info("Supervised result: {}".format(dumps(result, indent=4)))
    return result, model_best


def save_glove(glove_embs, glove_dir=cfg["paths"]["embedding_dir"][plat][user],
               glove_file='oov_glove.txt'):
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
                        default=cfg['sampling']['num_train_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['model']['use_cuda'], action='store_true')

    args = parser.parse_args()

    data_dir = cfg["paths"]["dataset_dir"][plat][user]

    train_df = read_labelled_json(data_dir, args.dataset_name)
    train_df = labels_mapper(train_df)

    test_df = read_labelled_json(data_dir, cfg['data']['target']['labelled'])

    result, model_outputs = BERT_classifier(
        train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
        model_name=args.model_name, model_type=args.model_type,
        num_epoch=args.num_train_epochs, use_cuda=True)
    exit(0)
    # cls_freqs = token_class_proba(df)
    # label_vec = token_dist2token_labels(cls_freq, vocab_set)
    #

    # glove_embs = glove2dict()
    # # df = labels_mapper(df)
    # train_portions = [0.2, 0.5, 1.0]
    # final_result = []
    # for train_portion in train_portions:
    #     s2i_dict, X_hat = main(glove_embs=glove_embs)
    #
    #     train, test = split_target(df, test_size=0.3,
    #     train_size=train_portion)
    #
    #     GCN_result = classify(test_df=test, stoi=s2i_dict, vectors=X_hat,
    #                           lr=1e-5, dropout=0.5)
    #
    #     glove_result = classify(train_df=train, test_df=test)
    #
    #     params = {
    #         'train_portion': train_portion,
    #     }
    #     result_dict = {
    #         'params': params,
    #         'glove':  glove_result,
    #         'gcn':    GCN_result
    #     }
    #     final_result.append(result_dict)
    #
    # logger.info(dumps(final_result, indent=4))
    #
    # exit(0)
    # result = load(open('Tweet_GCN_results.json'))
    # flatten_results(result)
    ## TODO:
    # 4. Ways to pretrain GCN:
    #   4.1 Domain classification
    #   4.2 Link prediction
    # 5. Restrict GCN propagation using class information
    # 6. Integrate various plotting functions
    # 7. Think about adversarial setting
    # 8. Use BERT for local embedding
    # 9. Concatenate Glove and GCN embedding and evaluate POC
    # 10. Think about pre-training GCN and GNN
    # 11. Add option to read hyper-params from config

    ## Generate embeddings for OOV tokens:
    glove_embs = glove2dict()

    epochs = [10, 25]
    layer_sizes = [1, 4]
    gcn_forward = [6, 2]
    hid_dims = [50, 100]
    dropouts = [0.5]
    lrs = [1e-5, 1e-6]

    final_result = []

    train_portions = [0.2, 0.5, 1.0]
    for train_portion in train_portions:
        train, test = split_target(df, test_size=0.3,
                                   train_size=train_portion)
        for c in gcn_forward:
            s2i_dict, X_hat = main(  # mittens_iter=20,
                gcn_hops=c, glove_embs=glove_embs)

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

                                logger.info("Result: {}".format(dumps(
                                    result_dict, indent=4)))

    logger.info("ALL Results: {}".format(dumps(final_result, indent=4)))

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
