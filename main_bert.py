# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Generate token graph
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

import random
import numpy as np
import argparse
from os import environ

import torch

from config import configuration as cfg, dataset_dir, platform as plat, pretrain_dir, username as user, cuda_device
from File_Handlers.csv_handler import read_csv, read_csvs
from stf_classification.BERT_multilabel_classifier import BERT_multilabel_classifier
from Logger.logger import logger

if torch.cuda.is_available() and cfg['cuda']['cuda_devices'][plat][user]:
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    torch.cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])
else:
    device_id = -1
# device_id, cuda_device = set_cuda_device()
device_id = cfg['cuda']['cuda_devices'][plat][user]


def set_all_seeds(seed=0):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_all_seeds(0)

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("-d", "--dataset_name",
                    default=cfg['data']['name'], type=str)
parser.add_argument("-m", "--model_name",
                    default=cfg['transformer']['model_name'], type=str)
parser.add_argument("-mt", "--model_type",
                    default=cfg['transformer']['model_type'], type=str)
parser.add_argument("-ne", "--num_train_epochs",
                    default=cfg['transformer']['num_epoch'], type=int)
parser.add_argument("-c", "--use_cuda",
                    default=cfg['cuda']['use_cuda'], action='store_true')

args = parser.parse_args()

# pepochs = cfg['pretrain']['epoch']
pepochs = [100, 50]
# for pepoch in pepochs:
if cfg['data']['zeroshot']:
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

BERT_multilabel_classifier(
    train_df=train_df, val_df=val_df, test_df=test_df,
    dataset_name=args.dataset_name, model_name=args.model_name,
    model_type=args.model_type, num_epoch=args.num_train_epochs,
    use_cuda=args.use_cuda, exp_name='BERT_GCPD_zeroshot_' + str(cfg['pretrain']['epoch']),
    pretrain_embs=True, pepoch=cfg['pretrain']['epoch'], run_cross_tests=False)

if cfg['data']['zeroshot']:
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

BERT_multilabel_classifier(
    train_df=train_df, val_df=val_df, test_df=test_df,
    dataset_name=args.dataset_name, model_name=args.model_name,
    model_type=args.model_type, num_epoch=args.num_train_epochs,
    use_cuda=args.use_cuda, exp_name='BERT_base_zeroshot',
    run_cross_tests=False)

# if __name__ == "__main__":
#
#     data_dir = dataset_dir
#
#     logger.info('Running GCPD.')
#     seed_count = cfg['training']['seed_count']
#     seed_start = cfg['training']['seed_start']
#     logger.info(f'Run for [{seed_count}] SEEDS')
#     for seed in range(seed_start, seed_start + seed_count):
#         logger.info(f'Setting SEED [{seed}]')
#         set_all_seeds(seed)
#         main_gcpd_1(glove_embs=glove_embs)
#
#     logger.info(f"Execution complete for {seed_count} SEEDs.")
