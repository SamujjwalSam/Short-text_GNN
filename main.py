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
import numpy as np
from torch import cuda, save, load
from torch.utils.data import DataLoader
from pandas import DataFrame
from networkx import adjacency_matrix
from os import environ
from os.path import join, exists
from json import dumps

from Label_Propagation_PyTorch.label_propagation import fetch_all_nodes, label_propagation
from Utils.utils import count_parameters, logit2label, sp_coo2torch_coo
from Layers.bilstm_classifiers import BiLSTM_Classifier
from Pretrain.pretrain import get_pretrain_artifacts, calculate_vocab_overlap
# from File_Handlers.csv_handler import read_csv, load_csvs
from File_Handlers.json_handler import save_json, read_json, read_labelled_json
# from File_Handlers.read_datasets import load_fire16, load_smerp17
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Data_Handlers.torchtext_handler import dataset2bucket_iter
from Data_Handlers.create_datasets import create_unlabeled_datasets, prepare_splitted_datasets
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Data_Handlers.instance_handler_dgl import Instance_Dataset_DGL
from Trainer.glen_trainer import GLEN_trainer
from Trainer.gat_trainer import GAT_BiLSTM_trainer
from Trainer.gcn_lstm_trainer import GCN_LSTM_trainer
from Trainer.lstm_trainer import LSTM_trainer
from Trainer.mlp_trainer import MLP_trainer
from Text_Encoder.finetune_static_embeddings import glove2dict, calculate_cooccurrence_mat,\
    train_mittens, preprocess_and_find_oov
from Trainer.trainer import trainer, predict_with_label
from Plotter.plot_functions import plot_training_loss
from Metrics.metrics import calculate_performance_pl
from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Logger.logger import logger

## Enable multi GPU cuda environment:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


def set_all_seeds(seed=0):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_all_seeds(1)

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


def main(model_type='LSTM', data_dir: str = dataset_dir, lr=cfg["model"]["optimizer"]["lr"],
         mittens_iter: int = 300, glove_embs=None,
         labelled_source_name: str = cfg['data']['train'],
         labelled_val_name: str = cfg['data']['val'],
         labelled_test_name: str = cfg['data']['test'], use_lpa=False):
    logger.critical(f'Current Learning Rate: [{lr}]')
    labelled_source_path = join(data_dir, labelled_source_name)
    logger.info('Read labelled data and prepare')
    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        get_iter=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        train_dataname=labelled_source_name, val_dataname=labelled_val_name,
        test_dataname=labelled_test_name)

    logger.info(f'Classifying examples using [{model_type}] model.')
    if model_type == 'MLP':
        logger.info('Using GAT model')
        train_epochs_output_dict, test_output = MLP_trainer(
            X, train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            epochs=cfg['training']['num_epoch'], lr=lr)

    elif model_type == 'LSTM':
        logger.info('Using LSTM model')
        train_epochs_output_dict, test_output = LSTM_trainer(
            train_dataloader, val_dataloader, test_dataloader, vectors=train_vocab.vocab.vectors,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            epochs=cfg['training']['num_epoch'], lr=lr)

    elif model_type == 'GAT':
        logger.info('Create GAT dataloader')
        train_dataloader, val_dataloader, test_dataloader = get_graph_dataloader(
            model_type, train_dataset, val_dataset, test_dataset, train_vocab,
            labelled_source_name=cfg['data']['train'], labelled_val_name=cfg['data']['val'],
            labelled_target_name=cfg['data']['test'])

        logger.info('Using GAT model')
        train_epochs_output_dict, test_output = GAT_BiLSTM_trainer(
            train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            num_heads=cfg['gnn_params']['num_heads'], epochs=cfg['training']['num_epoch'], lr=lr)

    else:
        logger.info(f'Creating token graph for model: [{model_type}]')
        unlabelled_source_name = cfg["data"]["source"]['unlabelled']
        # labelled_target_name: str = cfg['data']['test'],
        unlabelled_target_name = cfg["data"]["target"]['unlabelled']
        S_dataname = unlabelled_source_name + "_data.csv"
        T_dataname = unlabelled_target_name + "_data.csv"
        if glove_embs is None:
            glove_embs = glove2dict()
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

            S_dataset, (S_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=S_dataname)
            T_dataset, (T_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=T_dataname)
        else:
            C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
            T_dataset, T_fields, labelled_token2vec_map, s_lab_df =\
                create_unlabeled_datasets(
                    s_lab_df=None, data_dir=data_dir,
                    labelled_source_name=labelled_source_name,
                    unlabelled_source_name=unlabelled_source_name,
                    unlabelled_target_name=unlabelled_target_name)
            ## Save vocabs:
            save_json(C_vocab, labelled_source_path + 'C_vocab')
            save_json(S_vocab, labelled_source_path + 'S_vocab')
            save_json(T_vocab, labelled_source_path + 'T_vocab')
            save_json(labelled_token2vec_map, labelled_source_path + 'labelled_token2vec_map')

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

        g_ob = Token_Dataset_nx(corpus_toks, C_vocab, labelled_source_name,
                                S_vocab, T_vocab)
        g_ob.add_edge_weights()
        G = g_ob.G

        node_list = list(G.nodes)
        logger.info(f"Number of nodes {len(node_list)} and edges {len(G.edges)} in token graph")

        logger.info(f'Create new embeddings for OOV tokens')
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

        logger.info(f'Get adjacency matrix and node embeddings in same order:')
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

        if use_lpa:
            logger.info(f'Apply Label Propagation to get label vectors for unlabelled nodes:')
            label_proba_filename = join(data_dir, labelled_source_name + "_lpa_vecs.pt")
            if exists(label_proba_filename):
                lpa_vecs = torch.load(label_proba_filename)
            else:
                all_node_labels, labelled_masks = fetch_all_nodes(
                    node_list, labelled_token2vec_map, C_vocab['idx2str_map'],
                    # default_fill=[0.])
                    default_fill=[0., 0., 0., 0.])
                lpa_vecs = label_propagation(adj, all_node_labels, labelled_masks)
                torch.save(lpa_vecs, label_proba_filename)

            logger.info('Recalculate edge weights using LPA vectors:')
            g_ob.normalize_edge_weights(lpa_vecs)

            adj = adjacency_matrix(g_ob.G, nodelist=node_list, weight='weight')
            adj = sp_coo2torch_coo(adj)

        logger.info('Normalize token graph:')
        adj = g_ob.normalize_adj(adj)

        # ## Create label to propagated vector map:
        # logger.info('Create label to propagated vector map')
        # node_txt2label_vec = {}
        # for node_id in node_list:
        #     node_txt2label_vec[C_vocab['idx2str_map'][node_id]] =\
        #         lpa_vecs[node_id].tolist()
        # DataFrame.from_dict(node_txt2label_vec, orient='index').to_csv(labelled_source_name +
        # 'node_txt2label_vec.csv')

        logger.critical('BEFORE ---------------------------------------------')
        if model_type == 'GCN':
            logger.info('Using GAT model')
            train_epochs_output_dict, test_output = GCN_LSTM_trainer(
                adj, X, train_dataloader, val_dataloader, test_dataloader,
                in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
                epochs=cfg['training']['num_epoch'], lr=lr)

        else:
            train_epochs_output_dict, test_output = GLEN_trainer(
                adj, X, train_dataloader, val_dataloader, test_dataloader,
                in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
                num_heads=cfg['gnn_params']['num_heads'], epochs=cfg['training']['num_epoch'], lr=lr)

        # train_epochs_output_dict, test_output = GAT_BiLSTM_trainer(
        #     adj, X, train_dataloader, val_dataloader, test_dataloader,
        #     in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
        #     epochs=cfg['training']['num_epoch'], lr=lr, state=None)

    logger.critical('PRETRAIN ###########################################')

    pretrain_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['data']['name'])
    data_filename = cfg['data']['name']
    emb_filename = join(pretrain_dir, data_filename + "_pretrained_emb.pt")
    save_path = join(pretrain_dir, cfg['data']['name'] + '_model.pt')
    if exists(join(pretrain_dir, data_filename + '_joint_vocab.csv'))\
            and exists(emb_filename) and exists(save_path):
        # joint_vocab = read_json(join(pretrain_dir, data_filename + '_joint_vocab'))
        logger.info('Accessing token graph node embeddings:')
        X = load(emb_filename)
        state = torch.load(save_path)
    else:
        state, joint_vocab, pretrain_embs = get_pretrain_artifacts(
            epochs=cfg['pretrain']['epochs'])
        calculate_vocab_overlap(train_vocab, joint_vocab)
        # save_json(joint_vocab, labelled_source_path + '_joint_vocab', overwrite=True)
        logger.info('Get node embeddings from token graph:')
        X = g_ob.get_node_embeddings(oov_embs, glove_embs, train_vocab['idx2str_map'], pretrain_embs=pretrain_embs)
        # X = sp_coo2torch_coo(X)
        save(X, emb_filename)

        ## Plot pretrained embeddings:
        # C = set(glove_embs.keys()).intersection(set(pretrain_embs.keys()))
        # logger.debug(f'Common vocab size: {len(C)}')
        # words = ['nepal', 'italy', 'building', 'damage', 'kathmandu', 'water',
        #          'wifi', 'need', 'available', 'earthquake']
        # X_glove = {word: glove_embs[word] for word in words}
        # X_gcn = {word: pretrain_embs[word].detach().cpu().numpy() for word in words}
        # from Plotter.plot_functions import plot_vecs_color
        #
        # plot_vecs_color(tokens2vec=X_gcn, save_name='gcn_pretrained.pdf')
        # plot_vecs_color(tokens2vec=X_glove, save_name='glove_pretrained.pdf')
        # logger.debug(f'Word list: {words}')

    logger.critical('AFTER None **********************************************')
    if model_type == 'GCN':
        logger.info('Using GAT model')
        train_epochs_output_dict, test_output = GCN_LSTM_trainer(
            adj, X, train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            epochs=cfg['training']['num_epoch'], lr=lr)
    else:
        train_epochs_output_dict, test_output = GLEN_trainer(
            adj, X, train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            num_heads=cfg['gnn_params']['num_heads'], epochs=cfg['training']['num_epoch'], lr=lr)


def get_graph_dataloader(model_type, train_dataset, val_dataset, test_dataset,
                         train_vocab, labelled_source_name: str = cfg['data']['train'],
                         labelled_val_name: str = cfg['data']['val'],
                         labelled_target_name: str = cfg['data']['test'],
                         train_batch_size=cfg['training']['train_batch_size'],
                         test_batch_size=cfg['training']['eval_batch_size']):
    logger.info(f'Creating instance graph dataloader for {model_type} model.')
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

    return train_dataloader, val_dataloader, test_dataloader


def classify(train_df=None, test_df=None, stoi=None, vectors=None,
             n_classes=cfg['data']['num_classes'], dim=cfg['embeddings']['emb_dim'],
             data_dir=dataset_dir, train_filename=cfg['data']['train'],
             test_filename=cfg['data']['test'], cls_thresh=None,
             epoch=cfg['training']['num_epoch'], num_layers=cfg['lstm_params']['num_layers'],
             num_hidden_nodes=cfg['lstm_params']['hid_size'], dropout=cfg['model']['dropout'],
             default_thresh=0.5, lr=cfg['model']['optimizer']['lr'],
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
    lrs = [1e-3]
    for lr in lrs:
        main(lr=lr)
    logger.info("Execution complete.")
