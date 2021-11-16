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
import pandas as pd
import torch
import random
import numpy as np
from torch import cuda, save, load
from os import environ

from File_Handlers.csv_handler import read_csv, read_csvs
from stf_classification.BERT_binary_classifier import BERT_binary_classifier
from stf_classification.multilabel_classifier_custom import BERT_multilabel_classifier
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, cuda_device, device_id
from Logger.logger import logger
from Data_Handlers.create_datasets import prepare_splitted_datasets
from main import classifier, get_gcpd_embs, add_pretrained2vocab
from Text_Encoder.finetune_static_embeddings import glove2dict

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


def kd_bert(kd_df, test_name: str = cfg['data']['test']):
    model_name = f'GCPD-KD_BERT'
    logger.critical(f'{model_name}')

    # val_df = read_csv(data_dir=pretrain_dir, data_file=val_name)
    # val_df = val_df.sample(frac=1)
    test_df = read_csv(data_dir=pretrain_dir, data_file=test_name)
    test_df = test_df.sample(frac=1)

    ## Call BERT for distillation:
    logger.info(f'Running BERT for model {model_name}')
    BERT_multilabel_classifier(
        train_df=kd_df, test_df=test_df, exp_name=model_name)
    logger.info("KD_BERT results")


def portion_bert(train_df, test_name: str = cfg['data']['test']):
    # ====================================================================
    ## Run BERT with 50% original data:
    # train_portions = [0.8, 0.5, 0.2]
    # for portion in train_portions:
    # train_df = read_csv(data_dir=pretrain_dir, data_file=train_name)
    # train_df = train_df.sample(frac=0.5)

    # val_df = read_csv(data_dir=pretrain_dir, data_file=val_name)
    # val_df = val_df.sample(frac=1)
    test_df = read_csv(data_dir=pretrain_dir, data_file=test_name)
    test_df = test_df.sample(frac=1)
    # print(test_df.shape)

    ## Call BERT for distillation:
    model_name = f'BERT_train_portion_{0.5}'
    logger.info(f'Running BERT for model {model_name}')
    BERT_binary_classifier(train_df=train_df, test_df=test_df, epoch=1)
    logger.info("BERT_0.5 results")


from os.path import join, exists
from Data_Handlers.create_datasets import prepare_single_dataset
# from main import run_all_multi
from Trainer.multi_trainer import run_all_multi


def main_kd(
        glove_embs=None, model_type='multi', train_name: str = cfg['data']['name'],
        val_name: str = cfg['data']['val'], test_name: str = cfg['data']['test'],
        epoch=cfg['training']['num_epoch'], lr=1e-3):

    # ====================================================================
    kd_filename = train_name + f'_BiLSTMEmb_GCPD_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}_'\
                               f'{str(lr)}_epoch_{str(epoch)}_kd'

    if glove_embs is None:
        glove_embs = glove2dict()
    logger.critical('KD ##########')
    model_name = f'GCPD_{model_type}_kd'
    logger.critical(f'GCPD-KD ********** {model_name}')
    logger.critical(f'Learning Rate: [{lr}]')
    train_portion = 0.5

    train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    train_dataloader, val_dataloader, test_dataloader, _, train_df =\
        prepare_splitted_datasets(
            get_dataloader=True, dim=cfg['embeddings']['emb_dim'],
            data_dir=dataset_dir, train_dataname=train_name,
            val_dataname=val_name, test_dataname=test_name, zeroshot=False,
            train_portion=train_portion)

    # tr_freq = train_vocab.vocab.freqs.keys()
    # tr_v = train_vocab.vocab.itos
    # ts_freq = test_vocab.vocab.freqs.keys()
    # ts_v = test_vocab.vocab.itos
    # ov_freq = set(tr_freq).intersection(ts_freq)
    # ov_v = set(tr_v).intersection(ts_v)
    # logger.info(
    #     f'Vocab train freq: {len(tr_freq)}, itos: {len(tr_v)}, '
    #     f'test freq: {len(ts_freq)}, itos: {len(ts_v)} = '
    #     f'overlap freq: {len(ov_freq)}, itos: {len(ov_v)}')
    #
    # train_vocab_mod = {
    #     'freqs':       train_vocab.vocab.freqs.copy(),
    #     'str2idx_map': dict(train_vocab.vocab.stoi.copy()),
    #     'idx2str_map': train_vocab.vocab.itos.copy(),
    # }
    # # model_name = f'Glove_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}'
    # # logger.critical(f'GLOVE ^^^^^^^^^^ {model_name}')
    # # classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
    # #            train_vocab, train_dataset, val_dataset, test_dataset,
    # #            train_name, glove_embs, lr, model_name=model_name)
    #
    # pmodel_type = cfg['pretrain']['model_type']
    # logger.critical('PRETRAIN ##########')
    # token2idx_map, X = get_gcpd_embs(
    #     train_dataset, train_vocab_mod, glove_embs, train_name,
    #     epoch=cfg['pretrain']['epoch'], model_type=pmodel_type)
    #
    # train_dataset, val_dataset, test_dataset, train_vocab, val_vocab, test_vocab,\
    # train_dataloader, val_dataloader, test_dataloader, extra_df, train_df = \
    #     prepare_splitted_datasets(
    #     stoi=token2idx_map, vectors=X, get_dataloader=True,
    #     dim=cfg['embeddings']['emb_dim'], data_dir=dataset_dir,
    #     train_dataname=train_name, val_dataname=val_name,
    #     test_dataname=test_name, zeroshot=cfg['data']['zeroshot'],
    #         train_portion=train_portion)

    # extra_datafile = train_name + str(train_portion) + ".csv"
    # logger.warning(f'New train data {train_name} size {train_df.shape} for train_portion {train_portion}')
    # extra_df.to_csv(join(dataset_dir, extra_datafile))
    #
    # _, _, extra_dataloader = prepare_single_dataset(dataname=extra_datafile)
    #
    # extra_pretrained_tokens = set(token2idx_map.keys()) - set(train_vocab_mod['str2idx_map'].keys())
    # logger.info(f'Add {len(extra_pretrained_tokens)} extra pretrained vectors to vocab')
    # if len(extra_pretrained_tokens) > 0:
    #     train_vocab = add_pretrained2vocab(extra_pretrained_tokens, token2idx_map, X, train_vocab)
    #
    # model_name = f'GCPD_{model_type}_freq{cfg["data"]["min_freq"]}'\
    #              f'_lr{str(lr)}'
    # logger.critical(f'GCPD ********** {model_name}')
    # # classifier(model_type, train_dataloader, val_dataloader, test_dataloader,
    # #            train_vocab, train_dataset, val_dataset, test_dataset,
    # #            train_name, glove_embs, lr, model_name=model_name)
    #
    # train_epochs_output_dict = run_all_multi(
    #     train_dataloader, val_dataloader, test_dataloader, vectors=train_vocab.vocab.vectors,
    #     in_dim=cfg['embeddings']['emb_dim'], epoch=cfg['training']['num_epoch'],
    #     model_name=model_name, kd_dataloader=extra_dataloader)
    #
    # # ====================================================================
    # ## For Disaster datasets:
    # kd_filename = train_name + f'_BiLSTMEmb_GCPD_{model_type}_freq{cfg["data"]["min_freq"]}_lr{str(lr)}_'\
    #                            f'{str(lr)}_epoch_{str(epoch)}_kd'
    #
    # # ## For Sentiment datasets:
    # # kd_filename = train_name + f'_BiLSTMEmb_GCPD_disaster_{str(lr)}_epoch_10_kd'
    #
    # ## Merge train and predicted extra data
    # kd_df = read_csv(data_dir=pretrain_dir, data_file=kd_filename)
    # # kd_df = kd_df.sample(frac=1)
    # kd_df = kd_df.drop(columns=["trues", "logits0", "logits1"]).rename(
    #     columns={'preds_hard': 'labels'})
    # kd_train_df = pd.concat([train_df, kd_df])
    # kd_train_df = kd_train_df.sample(frac=1)
    #
    # kd_bert(kd_train_df)
    portion_bert(train_df)

    logger.info("Execution complete.")


def main_kd_bert(
        model_type='disaster', train_name: str = cfg['data']['name'],
        val_name: str = cfg['data']['val'], test_name: str = cfg['data']['test'],
        epoch=cfg['training']['num_epoch'], lr=1e-3):
    logger.critical('PRETRAIN ##########')
    model_name = f'GCPD_{model_type}'
    logger.critical(f'GCPD ********** {model_name}')

    train_df = read_csv(data_dir=pretrain_dir,
                        data_file=train_name + f'_BiLSTMEmb_GCPD_{model_type}_{str(lr)}_epoch_{str(epoch)}_kd')
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

    train_df = read_csv(data_dir=pretrain_dir,
                        data_file=train_name + f'_BiLSTMEmb_GloVe_{model_type}_{str(lr)}_epoch_{str(epoch)}_kd')
    train_df = train_df.sample(frac=0.5)

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
