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
from os import environ

from File_Handlers.csv_handler import read_csv, read_csvs
from stf_classification.multilabel_classifier_custom import BERT_multilabel_classifier
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
    # torch.backends.cudnn.deterministic = True


def main_kd(
        model_type='disaster', train_name: str = cfg['data']['train'],
        val_name: str = cfg['data']['val'], test_name: str = cfg['data']['test'],
        epoch=cfg['training']['num_epoch'], lr=1e-3):
    logger.critical('PRETRAIN ##########')
    model_name = f'GCPD_{model_type}'
    logger.critical(f'GCPD ********** {model_name}')

    train_df = read_csv(data_dir=pretrain_dir, data_file=train_name + f'_BiLSTMEmb_GCPD_{model_type}_{str(lr)}_epoch_{str(epoch)}_kd')
    train_df = train_df.sample(frac=1)

    val_df = read_csv(data_dir=pretrain_dir, data_file=val_name)
    val_df = val_df.sample(frac=1)
    test_df = read_csv(data_dir=pretrain_dir, data_file=test_name)
    test_df = test_df.sample(frac=1)

    ## Call BERT for distillation:
    model_name = 'KD_' + model_name
    logger.info(f'Running BERT for model {model_name}')
    BERT_multilabel_classifier(
        train_df=train_df, val_df=val_df, test_df=test_df, exp_name=model_name)

    # ====================================================================

    model_name = f'GloVe_{model_type}'
    logger.critical(f'GLOVE ^^^^^^^^^^ {model_name}')

    train_df = read_csv(data_dir=pretrain_dir, data_file=train_name + f'_BiLSTMEmb_GloVe_{model_type}_{str(lr)}_epoch_{str(epoch)}_kd')
    train_df = train_df.sample(frac=1)

    val_df = read_csv(data_dir=pretrain_dir, data_file=val_name)
    val_df = val_df.sample(frac=1)
    test_df = read_csv(data_dir=pretrain_dir, data_file=test_name)
    test_df = test_df.sample(frac=1)

    ## Call BERT for distillation:
    model_name = 'KD_' + model_name
    logger.info(f'Running BERT for model {model_name}')
    BERT_multilabel_classifier(
        train_df=train_df, val_df=val_df, test_df=test_df, exp_name=model_name)

    logger.info("Execution complete.")


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

    logger.info('Running GCPD.')
    seed_count = cfg['training']['seed_count']
    seed_start = cfg['training']['seed_start']
    logger.info(f'Run for [{seed_count}] SEEDS')
    for seed in range(seed_start, seed_start + seed_count):
        logger.info(f'Setting SEED [{seed}]')
        set_all_seeds(seed)
        main_kd()

    logger.info(f"Execution complete for {seed_count} SEEDs.")
