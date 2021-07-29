# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Generates word embeddings for text using transformer models
__description__ : Uses simpletransformers to represent text
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "01/05/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import torch
import timeit
import argparse
import numpy as np
import pandas as pd
from os.path import join
from os import environ
from json import dumps, dump
from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from File_Handlers.csv_handler import read_csv, read_csvs
from Text_Processesor.build_corpus_vocab import get_token_embedding
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, device_id
# from Metrics.metrics import calculate_performance_bin_sk
from Logger.logger import logger

if torch.cuda.is_available() and cfg['cuda']['cuda_devices'][plat][user]:
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    torch.cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])
else:
    device_id = -1
# device_id, cuda_device = set_cuda_device()
device_id = cfg['cuda']['cuda_devices'][plat][user]


def format_df_cls(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer. """
    df['labels'] = df[df.columns[1:]].values.tolist()
    df = df[['text', 'labels']].copy()
    return df


def get_token_representations(
        train_df: pd.core.frame.DataFrame,
        # val_df: pd.core.frame.DataFrame, test_df: pd.core.frame.DataFrame,
        # n_classes: int = cfg['data']['num_classes'],
        # dataset_name: str = cfg['data']['train'],
        model_name: str = cfg['transformer']['model_name'],
        model_type: str = cfg['transformer']['model_type'],
        # num_epoch: int = cfg['transformer']['num_epoch'],
        use_cuda: bool = cfg['cuda']['use_cuda'],
        exp_name='BERT_representation', train_all_bert=False, format_input=False) -> (dict, dict):
    """Train and Evaluation data needs to be in a Pandas Dataframe

    containing at least two columns, a 'text' and a 'labels' column. The
    `labels` column should contain multi-hot encoded lists.

    :param n_classes:
    :param test_df:
    :param train_df:
    :param dataset_name:
    :param model_name:
    :param model_type:
    :param num_epoch:
    :param use_cuda:
    :return:
    """
    if format_input:
        train_df = format_df_cls(train_df)
        # val_df = format_df_cls(val_df)
        # test_df = format_df_cls(test_df)

    ## Add arguments:
    model_args = ModelArgs(evaluate_during_training=True)
    # model_args.num_labels = n_classes
    model_args.no_cache = True
    model_args.no_save = True
    # model_args.num_train_epochs = num_epoch
    model_args.output_dir = cfg['paths']['result_dir']
    model_args.cache_dir = cfg['paths']['cache_dir']
    model_args.fp16 = False
    model_args.max_seq_length = cfg['transformer']['max_seq_len']
    model_args.train_batch_size = cfg['transformer']['train_batch_size']
    model_args.overwrite_output_dir = True
    model_args.eval_batch_size = cfg['transformer']['eval_batch_size']
    # model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_verbose = True
    # model_args.evaluate_during_training_silent = False
    model_args.evaluate_each_epoch = True
    model_args.use_early_stopping = True
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_optimizer_and_scheduler = False
    model_args.reprocess_input_data = True
    # model_args.evaluate_during_training_steps = 3000
    model_args.save_steps = 10000
    model_args.n_gpu = 1
    model_args.threshold = 0.5
    model_args.early_stopping_patience = 3
    if not train_all_bert:
        logger.warning(f'Training classifier only.')
        model_args.train_custom_parameters_only = True
        model_args.custom_parameter_groups = [
            {
                "params": ["classifier.weight"],
                "lr": 1e-3,
            },
            {
                "params": ["classifier.bias"],
                "lr": 1e-3,
                "weight_decay": 0.0,
            },
        ]

    """
    You can set config in predict(): 
        {"output_hidden_states": True} in model_args to get the hidden states.

    This will give you:
    
        all_embedding_outputs: Numpy array of shape (batch_size, sequence_length, hid_dim)
        
        all_layer_hidden_states: Numpy array of shape (num_hidden_layers, batch_size, sequence_length, hid_dim)
    """
    # model_args.output_hidden_states = True

    ## Create a MultiLabelClassificationModel
    if torch.cuda.is_available() and cfg['cuda']['cuda_devices'][plat][user]:
        model = RepresentationModel(
            model_type=model_type, model_name=model_name,
            use_cuda=use_cuda and torch.cuda.is_available(), args=model_args,
            cuda_device=device_id)
    else:
        model = RepresentationModel(model_type=model_type, model_name=model_name,
            use_cuda=False, args=model_args)

    word_vectors = model.encode_sentences(train_df["text"].to_list(),
                                          combine_strategy=None)

    return word_vectors


if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("-d", "--dataset_name",
    #                     default=cfg['data']['name'], type=str)
    # parser.add_argument("-m", "--model_name",
    #                     default=cfg['transformer']['model_name'], type=str)
    # parser.add_argument("-mt", "--model_type",
    #                     default=cfg['transformer']['model_type'], type=str)
    # parser.add_argument("-ne", "--num_train_epochs",
    #                     default=cfg['training']['num_epoch'], type=int)
    # parser.add_argument("-c", "--use_cuda",
    #                     default=cfg['cuda']['use_cuda'], action='store_true')
    #
    # args = parser.parse_args()

    train_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['train'])
    train_df = train_df.sample(frac=1)
    # train_df = train_df.sample(frac=1)
    # train_df["labels"] = pd.to_numeric(train_df["labels"], downcast="float")
    # val_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['val'])
    # val_df["labels"] = pd.to_numeric(val_df["labels"], downcast="float")
    # test_df = read_csv(data_dir=dataset_dir, data_file=cfg['data']['test'])
    # test_df = test_df.sample(frac=1)
    # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
    token_embs = get_token_representations(train_df)
