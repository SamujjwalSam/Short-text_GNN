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
from collections import Counter
from typing import Dict

# from Label_Propagation_PyTorch.label_propagation import fetch_all_nodes,
# label_propagation
from Utils.utils import count_parameters, logit2label, sp_coo2torch_coo,\
    get_token2pretrained_embs, merge_dicts, clean_dataset_dir
from Layers.bilstm_classifiers import BiLSTM_Classifier
from Pretrain.pretrain import get_pretrain_artifacts, calculate_vocab_overlap,\
    get_w2v_embs, get_cnlp_embs
from File_Handlers.csv_handler import read_csv, read_csvs
from File_Handlers.json_handler import save_json, read_json, read_labelled_json
# from File_Handlers.read_datasets import load_fire16, load_smerp17
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Data_Handlers.torchtext_handler import dataset2bucket_dataloader
from Data_Handlers.create_datasets import create_unlabeled_datasets, prepare_splitted_datasets,\
    prepare_BERT_splitted_datasets, prepare_single_dataset, split_csv_train_data,\
    get_BERT_LSTM_dataloader, prepare_example_contrast_datasets
from Text_Processesor.build_corpus_vocab import get_dataset_fields, get_token_embedding
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Data_Handlers.instance_handler_dgl import Instance_Dataset_DGL
from Trainer.glen_trainer import GLEN_trainer
from Trainer.gat_trainer import GAT_BiLSTM_trainer
# from Trainer.gcn_lstm_trainer import GCN_LSTM_trainer
from Trainer.bert_lstm_trainer import BERT_LSTM_trainer
from Trainer.lstm_trainer import LSTM_trainer
from Trainer.dpcnn_trainer import DPCNN_trainer
from Trainer.mlp_trainer import MLP_trainer
from Transformers_simpletransformers.BERT_multilabel_classifier import BERT_multilabel_classifier
from Text_Encoder.finetune_static_embeddings import glove2dict, get_oov_vecs,\
    train_mittens, preprocess_and_find_oov2, create_clean_corpus, get_oov_tokens
from Trainer.trainer import trainer, predict_with_label
# from Plotter.plot_functions import plot_training_loss
from Metrics.metrics import calculate_performance_pl
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, cuda_device, device_id
from Logger.logger import logger

if cuda.is_available() and cfg['cuda']["use_cuda"][plat][user]:
    environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    cuda.set_device(device_id)
    logger.debug(cuda_device)
    logger.debug(f'current_device: {torch.cuda.current_device()}\n'
                 f' device_count: {torch.cuda.device_count()}')

    if cuda_device.type == 'cuda':
        # logger.info(torch.cuda.get_device_name(cfg['cuda']['use_cuda']))
        logger.info(f'Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)}GB')
        logger.info(f'Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)}GB')


def set_all_seeds(seed=0):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab):
    """ Adds extra pretrained tokens and vectors to TorchText train vocab.

    NOTE: "extra" here means tokens which are not present in train data vocab.

    :param extra_pretrained_tokens:
    :param token2idx_map:
    :param X:
    :param train_vocab:
    :return:
    """
    extra_vecs = []
    extra_idx_start = len(train_vocab.vocab.itos)
    for token in extra_pretrained_tokens:
        extra_vecs.append(X[token2idx_map[token]])
        train_vocab.vocab.stoi.__setitem__(token, extra_idx_start)
        train_vocab.vocab.itos.append(token)
        extra_idx_start += 1
    extra_vecs = torch.stack(extra_vecs)
    train_vocab.vocab.vectors = torch.cat((train_vocab.vocab.vectors, extra_vecs), 0)

    return train_vocab


# logger.info('Plot pretrained embeddings:')
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


def main_glen(model_type='GLEN', glove_embs=None, train_name: str = cfg['data']['train'],
              val_name: str = cfg['data']['val'], test_name: str = cfg['data']['test'],
              train_portion=None):
    logger.info('running MAIN_GLEN')
    if train_portion is not None:
        split_csv_train_data(dataset_name=train_name,
                             dataset_dir=dataset_dir, frac=train_portion,
                             dataset_save_name=train_name + '.csv')
    logger.info('Read and prepare labelled data for Word level')
    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        train_dataname=train_name, val_dataname=val_name,
        test_dataname=test_name)

    if cfg['data']['use_all_data']:
        train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
        train_df = train_df.sample(frac=1)
        val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
        val_df = val_df.sample(frac=1)
        test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
        test_df = test_df.sample(frac=1)
        # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    else:
        train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['train'])
        train_df = train_df.sample(frac=1)
        # train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
        val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
        val_df = val_df.sample(frac=1)
        # val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
        test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
        test_df = test_df.sample(frac=1)
        # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    if glove_embs is None:
        glove_embs = glove2dict()

    lrs = cfg['model']['lrs']
    logger.info(f'Run for multiple LRs: {lrs}')
    for lr in lrs:
        logger.critical(f'Current Learning Rate: [{lr}]')

        model_name = f'{model_type}_portion{str(train_portion)}_lr{str(lr)}'
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        model_name = f'LSTM_portion{str(train_portion)}_lr{str(lr)}'
        classifier('LSTM', train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

    model_name = f'BERT_portion{str(train_portion)}'
    logger.info(f'Running BERT for model {model_name}')
    BERT_multilabel_classifier(
        train_df=train_df, val_df=val_df, test_df=test_df,
        exp_name=model_name)

    logger.info("Execution complete.")


def main_gcpd_normal(model_type=cfg['model']['type'], glove_embs=None,
                     train_name: str = cfg['data']['train'],
                     val_name: str = cfg['data']['val'],
                     test_name: str = cfg['data']['test']):
    if glove_embs is None:
        glove_embs = glove2dict()

    lrs = cfg['model']['lrs']
    logger.info(f'Run for multiple LRs {lrs}')
    for lr in lrs:
        logger.critical(f'Current Learning Rate: [{lr}]')

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        tr_freq = train_vocab.vocab.freqs.keys()
        tr_v = train_vocab.vocab.itos
        ts_freq = test_vocab.vocab.freqs.keys()
        ts_v = test_vocab.vocab.itos
        ov_freq = set(tr_freq).intersection(ts_freq)
        ov_v = set(tr_v).intersection(ts_v)
        logger.info(
            f'Vocab train freq: {len(tr_freq)}, itos: {len(tr_v)}, '
            f'test freq: {len(ts_freq)}, itos: {len(ts_v)} = '
            f'overlap freq: {len(ov_freq)}, itos: {len(ov_v)}')

        train_vocab_mod = {
            'freqs':       train_vocab.vocab.freqs.copy(),
            'str2idx_map': dict(train_vocab.vocab.stoi.copy()),
            'idx2str_map': train_vocab.vocab.itos.copy(),
        }
        model_name = f'Glove_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'GLOVE ^^^^^^^^^^ {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        pmodel_type = cfg['pretrain']['model_type']
        logger.critical('PRETRAIN ##########')
        token2idx_map, X = get_gcpd_embs(
            train_dataset, train_vocab_mod, glove_embs, train_name,
            epoch=cfg['pretrain']['epoch'], model_type=pmodel_type)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=cfg['data']['use_all_data'])

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)

        model_name = f'GCPD_{model_type}_freq{cfg["data"]["min_freq"]}'\
                     f'_lr{str(lr)}_Pepoch{str(cfg["pretrain"]["epoch"])}_Pmodel{pmodel_type}'
        logger.critical(f'GCPD ********** {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        token2idx_map, X = get_w2v_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=cfg['data']['use_all_data'])
        model_name = f'W2V_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'WORD2VEC @@@@@@@@@@ {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        token2idx_map, X = get_cnlp_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=cfg['data']['use_all_data'])
        model_name = f'cnlp_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'cnlp &&&&&&&&&& {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

    # logger.info('Read and prepare labelled data for BERT')
    # train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['train'])
    # train_df = train_df.sample(frac=1)
    # val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    # val_df = val_df.sample(frac=1)
    # test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    # test_df = test_df.sample(frac=1)
    # # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    #
    # bert_full_mode = True
    # model_name = f'BERT_full{str(bert_full_mode)}'
    # logger.info(f'Running BERT for experiment: {model_name}')
    # BERT_multilabel_classifier(
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     exp_name=model_name, train_all_bert=bert_full_mode)
    # bert_full_mode = False
    # model_name = f'BERT_full{str(bert_full_mode)}'
    # logger.info(f'Running BERT for experiment: {model_name}')
    # BERT_multilabel_classifier(
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     exp_name=model_name, train_all_bert=bert_full_mode, format_input=False)

    logger.info("Execution complete.")


def main_gcpd_zeroshot(model_type='LSTM', glove_embs=None, train_name=cfg['data']['train'],
                       val_name=cfg['data']['val'], test_name=cfg['data']['test']):
    # if cfg['data']['use_all_data']:
    #     train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
    #     train_df = train_df.sample(frac=1)
    #     val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    #     val_df = val_df.sample(frac=1)
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    # else:
    #     train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['train'])
    #     train_df = train_df.sample(frac=1)
    #     # train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
    #     val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    #     val_df = val_df.sample(frac=1)
    #     # val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    # train_dataset, train_dataloader = get_BERT_LSTM_dataloader(train_df)
    # val_dataset, val_dataloader = get_BERT_LSTM_dataloader(val_df)
    # test_dataset, test_dataloader = get_BERT_LSTM_dataloader(test_df)

    # model_type = 'BERT_LSTM'
    # logger.info('Run for multiple LR')
    # lrs = cfg['model']['lrs']
    # for lr in lrs:
    #     model_name = f'BERT_LSTM_zeroshot_{model_type}_lr{str(lr)}'
    #     logger.critical(f'BERT_LSTM ********** {model_name}')
    #     classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
    #                None, train_dataset, val_dataset, test_dataset,
    #                train_name, glove_embs, lr, model_name=model_name)
    #
    # model_name = f'BERT_portion{str(0.9999)}'
    # logger.info(f'Running BERT for model {model_name}')
    # BERT_multilabel_classifier(
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     exp_name=model_name)

    examcon_dataset, examcon_dataloader = prepare_example_contrast_datasets(train_name)
    logger.info('Read and prepare labelled data for Word level')
    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        train_dataname=train_name, val_dataname=val_name, test_dataname=test_name,
        use_all_data=True)

    tr_freq = train_vocab.vocab.freqs.keys()
    tr_v = train_vocab.vocab.itos
    ts_freq = test_vocab.vocab.freqs.keys()
    ts_v = test_vocab.vocab.itos
    ov_freq = set(tr_freq).intersection(ts_freq)
    ov_v = set(tr_v).intersection(ts_v)
    logger.info(
        f'Vocab train freq: {len(tr_freq)}, itos: {len(tr_v)}, '
        f'test freq: {len(ts_freq)}, itos: {len(ts_v)} = '
        f'overlap freq: {len(ov_freq)}, itos: {len(ov_v)}')

    train_vocab_mod = {
        'freqs':       train_vocab.vocab.freqs.copy(),
        'str2idx_map': dict(train_vocab.vocab.stoi.copy()),
        'idx2str_map': train_vocab.vocab.itos.copy(),
    }

    if glove_embs is None:
        glove_embs = glove2dict()

    logger.info('Run for multiple LR')
    lrs = cfg['model']['lrs']
    for lr in lrs:
        logger.critical(f'Current Learning Rate: [{lr}]')
        pmodel_type = cfg['pretrain']['model_type']
        logger.critical('PRETRAIN ##########')
        token2idx_map, X = get_gcpd_embs(
            train_dataset, train_vocab_mod, glove_embs, train_name,
            epoch=cfg['pretrain']['epoch'], model_type=pmodel_type)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=True)

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)

        model_name = f'GCPD_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}'\
                     f'_lr{str(lr)}_Pepoch{str(cfg["pretrain"]["epoch"])}_Pmodel{pmodel_type}'
        logger.critical(f'GCPD ********** {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        token2idx_map, X = get_w2v_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=True)
        model_name = f'W2V_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'WORD2VEC @@@@@@@@@@ {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        token2idx_map, X = get_cnlp_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=True)
        model_name = f'cnlp_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'cnlp &&&&&&&&&& {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name, test_dataname=test_name,
            use_all_data=True)
        model_name = f'Glove_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'GLOVE ^^^^^^^^^^ {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name)

    logger.info("Execution complete.")


def main_gcpd_zeroshot_examcon(model_type='LSTM', glove_embs=None, train_name=cfg['data']['train'],
                               val_name=cfg['data']['val'], test_name=cfg['data']['test']):
    # if cfg['data']['use_all_data']:
    #     train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
    #     train_df = train_df.sample(frac=1)
    #     val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    #     val_df = val_df.sample(frac=1)
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    # else:
    #     train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['train'])
    #     train_df = train_df.sample(frac=1)
    #     # train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
    #     val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    #     val_df = val_df.sample(frac=1)
    #     # val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
    #     test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    #     test_df = test_df.sample(frac=1)
    #     # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")

    # train_dataset, train_dataloader = get_BERT_LSTM_dataloader(train_df)
    # val_dataset, val_dataloader = get_BERT_LSTM_dataloader(val_df)
    # test_dataset, test_dataloader = get_BERT_LSTM_dataloader(test_df)

    # model_type = 'BERT_LSTM'
    # logger.info('Run for multiple LR')
    # lrs = cfg['model']['lrs']
    # for lr in lrs:
    #     model_name = f'BERT_LSTM_zeroshot_{model_type}_lr{str(lr)}'
    #     logger.critical(f'BERT_LSTM ********** {model_name}')
    #     classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
    #                None, train_dataset, val_dataset, test_dataset,
    #                train_name, glove_embs, lr, model_name=model_name)
    #
    # model_name = f'BERT_portion{str(0.9999)}'
    # logger.info(f'Running BERT for model {model_name}')
    # BERT_multilabel_classifier(
    #     train_df=train_df, val_df=val_df, test_df=test_df,
    #     exp_name=model_name)

    examcon_dataset, examcon_dataloader = prepare_example_contrast_datasets(train_name)
    logger.info('Read and prepare labelled data for Word level')
    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        train_dataname=train_name, val_dataname=val_name, test_dataname=test_name,
        use_all_data=True)

    # classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
    #            train_vocab, train_dataset, val_dataset, test_dataset,
    #            train_name, glove_embs, 1e-4, model_name='examcon',
    #            pretrain_dataloader=(examcon_dataset, examcon_dataloader))

    tr_freq = train_vocab.vocab.freqs.keys()
    tr_v = train_vocab.vocab.itos
    ts_freq = test_vocab.vocab.freqs.keys()
    ts_v = test_vocab.vocab.itos
    ov_freq = set(tr_freq).intersection(ts_freq)
    ov_v = set(tr_v).intersection(ts_v)
    logger.info(
        f'Vocab train freq: {len(tr_freq)}, itos: {len(tr_v)}, '
        f'test freq: {len(ts_freq)}, itos: {len(ts_v)} = '
        f'overlap freq: {len(ov_freq)}, itos: {len(ov_v)}')

    train_vocab_mod = {
        'freqs':       train_vocab.vocab.freqs.copy(),
        'str2idx_map': dict(train_vocab.vocab.stoi.copy()),
        'idx2str_map': train_vocab.vocab.itos.copy(),
    }

    if glove_embs is None:
        glove_embs = glove2dict()

    logger.info('Run for multiple LR')
    lrs = cfg['model']['lrs']
    for lr in lrs:
        logger.critical(f'Current Learning Rate: [{lr}]')
        pmodel_type = cfg['pretrain']['model_type']
        logger.critical('PRETRAIN ##########')
        token2idx_map, X = get_gcpd_embs(
            train_dataset, train_vocab_mod, glove_embs, train_name,
            epoch=cfg['pretrain']['epoch'], model_type=pmodel_type)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=True)

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)

        model_name = f'GCPD_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}'\
                     f'_lr{str(lr)}_Pepoch{str(cfg["pretrain"]["epoch"])}_Pmodel{pmodel_type}'
        logger.critical(f'GCPD ********** {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=(examcon_dataset, examcon_dataloader))

        # # ======================================================================
        #
        # token2idx_map, X = get_w2v_embs(glove_embs)
        # train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        # train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        #     stoi=token2idx_map, vectors=X, get_dataloader=True,
        #     dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        #     train_dataname=train_name, val_dataname=val_name,
        #     test_dataname=test_name, use_all_data=True)
        # model_name = f'W2V_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        # logger.critical(f'WORD2VEC @@@@@@@@@@ {model_name}')
        # classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
        #            train_vocab, train_dataset, val_dataset, test_dataset,
        #            train_name, glove_embs, lr, model_name=model_name)
        #
        # # ======================================================================
        #
        # token2idx_map, X = get_cnlp_embs(glove_embs)
        # train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        # train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
        #     stoi=token2idx_map, vectors=X, get_dataloader=True,
        #     dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
        #     train_dataname=train_name, val_dataname=val_name,
        #     test_dataname=test_name, use_all_data=True)
        # model_name = f'cnlp_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        # logger.critical(f'cnlp &&&&&&&&&& {model_name}')
        # classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
        #            train_vocab, train_dataset, val_dataset, test_dataset,
        #            train_name, glove_embs, lr, model_name=model_name)

        # ======================================================================

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name, test_dataname=test_name,
            use_all_data=True)
        model_name = f'Glove_zeroshot_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
        logger.critical(f'GLOVE ^^^^^^^^^^ {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=(examcon_dataset, examcon_dataloader))

    logger.info("Execution complete.")


def main_gcpd_alltrain(model_type=cfg['model']['type'], glove_embs=None,
                       train_name: str = cfg['data']['train'],
                       val_name: str = cfg['data']['val'],
                       test_name: str = cfg['data']['test'],
                       use_alltrain_vocab=True):
    if glove_embs is None:
        glove_embs = glove2dict()

    alltrain_datafile = "training_" + str(len(cfg['pretrain']['files'])) + ".csv"
    # if not exists(join(dataset_dir, train_dataname)):
    alltrain_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
    alltrain_df.to_csv(join(data_dir, alltrain_datafile))

    lrs = cfg['model']['lrs']
    logger.info(f'Run for multiple LR {lrs}')
    for lr in lrs:
        # logger.info('Read and prepare labelled data for Word level')
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            get_dataloader=True, dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        logger.info('Use pretrain vocab for alltrain and train both')
        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        model_name = f'Glove_alltrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}'
        logger.critical(f'Glove ********** {model_name}')
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

        tr_freq = train_vocab.vocab.freqs.keys()
        tr_v = train_vocab.vocab.itos
        ts_freq = test_vocab.vocab.freqs.keys()
        ts_v = test_vocab.vocab.itos
        ov_freq = set(tr_freq).intersection(ts_freq)
        ov_v = set(tr_v).intersection(ts_v)
        logger.info(
            f'Vocab train freq: {len(tr_freq)}, itos: {len(tr_v)}, '
            f'test freq: {len(ts_freq)}, itos: {len(ts_v)} = '
            f'overlap freq: {len(ov_freq)}, itos: {len(ov_v)}')

        train_vocab_mod = {
            'freqs':       train_vocab.vocab.freqs.copy(),
            'str2idx_map': dict(train_vocab.vocab.stoi.copy()),
            'idx2str_map': train_vocab.vocab.itos.copy(),
        }

        pmodel_type = cfg['pretrain']['model_type']
        logger.critical('PRETRAIN ##########')

        token2idx_map, X = get_gcpd_embs(
            train_dataset, train_vocab_mod, glove_embs, train_name,
            epoch=cfg['pretrain']['epoch'], model_type=pmodel_type)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            stoi=token2idx_map, vectors=X, dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)
            logger.info(f'Added {len(extra_pretrained_tokens)} extra tokens to vocab')

        model_name = f'GCPD_allpretrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}_glove_gcpd'

        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            stoi=token2idx_map, vectors=X, dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)
            logger.info(f'Added {len(extra_pretrained_tokens)} extra tokens to vocab')

        model_name = f'GCPD_allpretrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}_gcpd_gcpd'

        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
        logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
        if len(extra_pretrained_tokens) > 0:
            train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)
            logger.info(f'Added {len(extra_pretrained_tokens)} extra tokens to vocab')

        model_name = f'GCPD_allpretrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}_gcpd_glove'

        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

        # ======================================================================

        token2idx_map, X = get_w2v_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            stoi=token2idx_map, vectors=X, dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        model_name = f'W2V_allpretrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}'
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

        # ======================================================================

        token2idx_map, X = get_cnlp_embs(glove_embs)
        train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
        train_dataloader, val_dataloader, test_dataloader = prepare_splitted_datasets(
            stoi=token2idx_map, vectors=X, get_dataloader=True,
            dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
            train_dataname=train_name, val_dataname=val_name,
            test_dataname=test_name, use_all_data=False)

        alltrain_dataset, alltrain_vocab, alltrain_dataloader = prepare_single_dataset(
            stoi=token2idx_map, vectors=X, dataname=alltrain_datafile)

        if use_alltrain_vocab:
            train_vocab = alltrain_vocab

        model_name = f'cnlp_allpretrain_lr{str(lr)}_vocab{str(use_alltrain_vocab)}'
        classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
                   train_vocab, train_dataset, val_dataset, test_dataset,
                   train_name, glove_embs, lr, model_name=model_name,
                   pretrain_dataloader=alltrain_dataloader)

    logger.info("Execution complete.")


def get_gcpd_embs(train_dataset, train_vocab: Dict[str, Counter],
                  glove_embs: Dict[str, np.ndarray], train_name: str,
                  epoch: int = cfg['pretrain']['epoch'],
                  model_type=cfg['pretrain']['model_type']) -> (Dict, torch.tensor):
    """

    :rtype: dict, tensor
    """
    pretrain_vocab, pretrain_embs, X = get_pretrain_artifacts(
        epoch=epoch, model_type=model_type)
    calculate_vocab_overlap(set(train_vocab['str2idx_map'].keys()),
                            set(pretrain_vocab['str2idx_map'].keys()))

    if X is None:
        logger.info('Get token embeddings with pretrained vectors')
        high_oov, low_glove, low_oov, corpus, corpus_toks = get_oov_tokens(
            train_dataset, train_name, data_dir, train_vocab, glove_embs)
        oov_embs = get_oov_vecs(list(high_oov.keys()), corpus,
                                train_name, data_dir, glove_embs)
        X, _ = get_token_embedding(list(pretrain_vocab['str2idx_map'].keys(
        )), oov_embs, glove_embs, pretrain_embs)

    return pretrain_vocab['str2idx_map'], X


def classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
               train_vocab, train_dataset, val_dataset, test_dataset, dataname,
               glove_embs=None, lr=cfg['model']['optimizer']['lr'], model_name=None,
               pretrain_dataloader=None):
    if model_name is None:
        model_name = f'{model_type}_epoch{str(cfg["training"]["num_epoch"])}_lr{str(lr)}'
    logger.info(f'Classifying examples using [{model_type}] model.')
    datapath = join(data_dir, dataname)
    if model_type == 'MLP':
        train_epochs_output_dict, test_output = MLP_trainer(
            train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            epoch=cfg['training']['num_epoch'], lr=lr, model_name=model_name)

    # elif model_type == 'LSTM':
    #     epochs = cfg['training']['cls_pretrain_epochs']
    #     for epoch in epochs:
    #         train_epochs_output_dict = LSTM_trainer(
    #             train_dataloader, val_dataloader, test_dataloader, vectors=train_vocab.vocab.vectors,
    #             in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
    #             epoch=cfg['training']['num_epoch'], lr=lr, model_name=model_name,
    #             pretrain_dataloader=pretrain_dataloader,
    #             pretrain_epoch=epoch)

    elif model_type == 'LSTM':
        train_epochs_output_dict = LSTM_trainer(
            train_dataloader, val_dataloader, test_dataloader, vectors=train_vocab.vocab.vectors,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            epoch=cfg['training']['num_epoch'], lr=lr, model_name=model_name,
            pretrain_dataloader=pretrain_dataloader)

    elif model_type == 'BERT_LSTM':
        train_epochs_output_dict = BERT_LSTM_trainer(
            train_dataloader, val_dataloader, test_dataloader,
            in_dim=train_dataloader.dataset[0][0].shape[1],
            hid_dim=cfg['gnn_params']['hid_dim'],
            epoch=cfg['training']['num_epoch'], lr=lr, model_name=model_name,
            # pretrain_dataloader=pretrain_dataloader, pretrain_epoch=epoch
        )

    elif model_type == 'GAT':
        logger.info('Create GAT dataloader')
        train_dataloader, val_dataloader, test_dataloader = get_graph_dataloader(
            model_type, train_dataset, val_dataset, test_dataset, train_vocab,
            train_name=cfg['data']['train'], val_name=cfg['data']['val'],
            labelled_target_name=cfg['data']['test'])

        train_epochs_output_dict, test_output = GAT_BiLSTM_trainer(
            train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            num_heads=cfg['gnn_params']['num_heads'], epoch=cfg['training']['num_epoch'],
            lr=lr, model_name=model_name)

    else:  ## GLEN
        logger.info(f'Creating token graph for model: [{model_type}]')
        untrain_name = cfg["data"]["source"]['unlabelled']
        # labelled_target_name: str = cfg['data']['test'],
        unlabelled_target_name = cfg["data"]["target"]['unlabelled']
        # S_dataname = untrain_name + "_data.csv"
        # T_dataname = unlabelled_target_name + "_data.csv"
        if glove_embs is None:
            glove_embs = glove2dict()
        if exists(datapath + 'S_vocab.json')\
                and exists(datapath + 'T_vocab.json')\
                and exists(datapath + 'labelled_token2vec_map.json'):
            # ## Read labelled source data
            # s_lab_df = read_labelled_json(data_dir, train_name)
            # ## Match label space between two datasets:
            # if str(train_name).startswith('fire16'):
            #     s_lab_df = labels_mapper(s_lab_df)

            C_vocab = read_json(datapath + 'C_vocab')
            S_vocab = read_json(datapath + 'S_vocab')
            T_vocab = read_json(datapath + 'T_vocab')
            labelled_token2vec_map = read_json(datapath + 'labelled_token2vec_map')

            S_dataset, (S_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=untrain_name + ".csv")
            T_dataset, (T_fields, LABEL) = get_dataset_fields(
                csv_dir=data_dir, csv_file=unlabelled_target_name + ".csv")
        else:
            C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
            T_dataset, T_fields, labelled_token2vec_map, s_lab_df =\
                create_unlabeled_datasets(
                    s_lab_df=None, data_dir=data_dir, train_name=dataname,
                    untrain_name=untrain_name,
                    unlabelled_target_name=unlabelled_target_name)
            ## Save vocabs:
            save_json(C_vocab, datapath + 'C_vocab')
            save_json(S_vocab, datapath + 'S_vocab')
            save_json(T_vocab, datapath + 'T_vocab')
            save_json(labelled_token2vec_map, datapath + 'labelled_token2vec_map')

        if exists(datapath + 'S_corpus.json')\
                and exists(datapath + 'T_corpus.json')\
                and exists(datapath + 'S_corpus_toks.json')\
                and exists(datapath + 'T_corpus_toks.json'):
            # S_high_oov = read_json(datapath + 'S_high_oov')
            # T_high_oov = read_json(datapath + 'T_high_oov')
            # low_glove = read_json(train_name+'_low_glove')
            S_corpus = read_json(datapath + 'S_corpus', convert_ordereddict=False)
            T_corpus = read_json(datapath + 'T_corpus', convert_ordereddict=False)
            S_corpus_toks = read_json(datapath + 'S_corpus_toks', convert_ordereddict=False)
            T_corpus_toks = read_json(datapath + 'T_corpus_toks', convert_ordereddict=False)
            S_high_oov = read_json(datapath + 'S_high_oov', convert_ordereddict=False)
            T_high_oov = read_json(datapath + 'T_high_oov', convert_ordereddict=False)
        else:
            ## Get all OOVs which does not have Glove embedding:
            # high_oov, low_glove, corpus, corpus_toks =\
            S_high_oov, S_low_glove, S_low_oov = preprocess_and_find_oov2(C_vocab, glove_embs=glove_embs,
                                                                          labelled_vocab_set=set(
                                                                              labelled_token2vec_map.keys()))
            S_corpus, S_corpus_toks, _ = create_clean_corpus(S_dataset, S_low_oov)

            T_high_oov, T_low_glove, T_low_oov =\
                preprocess_and_find_oov2(C_vocab, glove_embs=glove_embs,
                                         labelled_vocab_set=set(labelled_token2vec_map.keys()))
            T_corpus, T_corpus_toks, _ = create_clean_corpus(T_dataset, T_low_oov)

            ## Save token sets: high_oov, low_glove, corpus, corpus_toks
            save_json(S_high_oov, datapath + 'S_high_oov')
            save_json(T_high_oov, datapath + 'T_high_oov')
            # save_json(low_glove, train_name+'_low_glove', overwrite=True)
            save_json(S_corpus, datapath + 'S_corpus')
            save_json(T_corpus, datapath + 'T_corpus')
            save_json(S_corpus_toks, datapath + 'S_corpus_toks')
            save_json(T_corpus_toks, datapath + 'T_corpus_toks')
            save_json(C_vocab, datapath + 'C_vocab', overwrite=True)

        # high_oov, low_glove, low_oov, corpus, corpus_toks = get_oov_tokens(
        #     (S_dataset, T_dataset), dataname, data_dir, C_vocab, glove_embs)

        logger.info(f'Create new embeddings for OOV tokens')
        oov_emb_filename = dataname + '_OOV_vectors_dict'
        if exists(join(data_dir, oov_emb_filename + '.pkl')):
            logger.info('Read OOV embeddings:')
            oov_embs = load_pickle(filepath=data_dir, filename=oov_emb_filename)
        else:
            logger.info('Create OOV embeddings using Mittens:')
            # high_oov = S_high_oov + T_high_oov
            high_oov = merge_dicts(S_high_oov, T_high_oov)

            high_oov_tokens_list = list(high_oov.keys())
            c_corpus = S_corpus + T_corpus
            # oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, c_corpus)
            # oov_embs = train_mittens(oov_mat_coo, high_oov_tokens_list, glove_embs, max_iter=mittens_iter)
            # save_pickle(oov_embs, filepath=data_dir, filename=oov_emb_filename, overwrite=True)
            oov_embs = get_oov_vecs(high_oov_tokens_list, c_corpus, dataname, data_dir, glove_embs)

        g_ob = Token_Dataset_nx((S_corpus_toks, T_corpus_toks), C_vocab, dataname, S_vocab, T_vocab)
        g_ob.add_edge_weights()
        G = g_ob.G
        node_list = list(G.nodes)
        logger.info(f"Number of nodes {len(node_list)} and edges {len(G.edges)} in token graph")

        logger.info(f'Get adjacency matrix and node embeddings in same order:')
        ## Note: Saving sparse tensor usually gets corrupted.
        # adj_filename = join(data_dir, train_name + "_adj.pt")
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
        emb_filename = join(data_dir, dataname + "_emb.pt")
        if exists(emb_filename):
            X = load(emb_filename)
        else:
            logger.info('Get node embeddings from token graph:')
            X = g_ob.get_node_embeddings(oov_embs, glove_embs, C_vocab['idx2str_map'])
            # X = sp_coo2torch_coo(X)
            save(X, emb_filename)

        # if use_lpa:
        #     logger.info(f'Apply Label Propagation to get label vectors for unlabelled nodes:')
        #     label_proba_filename = join(data_dir, dataname + "_lpa_vecs.pt")
        #     if exists(label_proba_filename):
        #         lpa_vecs = torch.load(label_proba_filename)
        #     else:
        #         all_node_labels, labelled_masks = fetch_all_nodes(
        #             node_list, labelled_token2vec_map, C_vocab['idx2str_map'],
        #             # default_fill=[0.])
        #             default_fill=[0., 0., 0., 0.])
        #         lpa_vecs = label_propagation(adj, all_node_labels, labelled_masks)
        #         torch.save(lpa_vecs, label_proba_filename)
        #
        #     logger.info('Recalculate edge weights using LPA vectors:')
        #     g_ob.normalize_edge_weights(lpa_vecs)
        #
        #     adj = adjacency_matrix(g_ob.G, nodelist=node_list, weight='weight')
        #     adj = sp_coo2torch_coo(adj)

        logger.info('Normalize token graph:')
        adj = g_ob.normalize_adj(adj)

        # ## Create label to propagated vector map:
        # logger.info('Create label to propagated vector map')
        # node_txt2label_vec = {}
        # for node_id in node_list:
        #     node_txt2label_vec[C_vocab['idx2str_map'][node_id]] =\
        #         lpa_vecs[node_id].tolist()
        # DataFrame.from_dict(node_txt2label_vec, orient='index').to_csv(train_name +
        # 'node_txt2label_vec.csv')

        logger.info('Get graph dataloader')
        train_dataloader, val_dataloader, test_dataloader = get_graph_dataloader(
            model_type, train_dataset, val_dataset, test_dataset, train_vocab,
            train_name=cfg['data']['train'], val_name=cfg['data']['val'],
            labelled_target_name=cfg['data']['test'])

        train_epochs_output_dict = GLEN_trainer(
            adj, X, train_dataloader, val_dataloader, test_dataloader,
            in_dim=cfg['embeddings']['emb_dim'], hid_dim=cfg['gnn_params']['hid_dim'],
            num_heads=cfg['gnn_params']['num_heads'], epoch=cfg['training']['num_epoch'],
            lr=lr, model_name=model_name)


def get_graph_dataloader(model_type, train_dataset, val_dataset, test_dataset,
                         train_vocab, train_name: str = cfg['data']['train'],
                         val_name: str = cfg['data']['val'],
                         labelled_target_name: str = cfg['data']['test'],
                         train_batch_size=cfg['training']['train_batch_size'],
                         test_batch_size=cfg['training']['eval_batch_size']):
    logger.info(f'Creating instance graph dataloader for {model_type} model.')
    train_instance_graphs = Instance_Dataset_DGL(
        train_dataset, train_vocab, train_name, class_names=cfg[
            'data']['class_names'])
    logger.debug(train_instance_graphs.num_labels)
    # logger.debug(train_instance_graphs.graphs, train_instance_graphs.labels)

    train_dataloader = DataLoader(train_instance_graphs, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=train_instance_graphs.batch_graphs)

    logger.info(f"Number of training instance graphs: {len(train_instance_graphs)}")

    val_instance_graphs = Instance_Dataset_DGL(
        val_dataset, train_vocab, val_name, class_names=cfg[
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

    logger.info('Get iterator')
    train_dataloader, val_dataloader = dataset2bucket_dataloader(
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
        model, train_dataloader, val_dataloader, N_EPOCHS=epoch, lr=lr)

    # plot_training_loss(losses['train'], losses['val'],
    #                    plot_name='loss' + str(epoch) + str(lr))

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
                          epoch=5, cls_thresh=None, n_classes=cfg['data']['num_classes']):
    """ Train and Predict on full supervised mode.

    Returns:

    """

    model_best, val_preds_trues_best, val_preds_trues_all, losses = trainer(
        model, train_iterator, val_iterator, N_EPOCHS=epoch)

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
    #                     default=cfg['cuda']['use_cuda'], action='store_true')
    #
    # args = parser.parse_args()

    data_dir = dataset_dir

    glove_embs = glove2dict()

    # logger.info('Running GLEN.')
    # train_portions = cfg['data']['train_portions']
    # seed_num = 3
    # seed_start = 0
    # logger.info(f'Run for [{seed_num}] SEEDS')
    # for train_portion in train_portions:
    #     clean_dataset_dir()
    #     logger.info(f'Run for train_portion: [{train_portion}]')
    #     for seed in range(seed_start, seed_start+seed_num+1):
    #         logger.info(f'Setting SEED [{seed}]')
    #         set_all_seeds(seed)
    #         main_glen(glove_embs=glove_embs, train_portion=train_portion)

    logger.info('Running GCPD.')
    seed_count = cfg['training']['seed_count']
    seed_start = cfg['training']['seed_start']
    logger.info(f'Run for [{seed_count}] SEEDS')
    for seed in range(seed_start, seed_start + seed_count):
        logger.info(f'Setting SEED [{seed}]')
        set_all_seeds(seed)
        # main_gcpd_alltrain(glove_embs=glove_embs)
        # main_gcpd_zeroshot(glove_embs=glove_embs)
        main_gcpd_zeroshot_examcon(glove_embs=glove_embs)

    logger.info(f"Execution complete for {seed_count} SEEDs.")
