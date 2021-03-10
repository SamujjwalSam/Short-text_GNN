# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Creates TorchText dataset
__description__ : Creates TorchText dataset
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "05/08/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from os.path import join, exists

from Text_Processesor.tokenizer import BERT_tokenizer
from File_Handlers.csv_handler import read_csv, read_csvs
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from File_Handlers.json_handler import save_json, read_json, read_labelled_json
from File_Handlers.read_datasets import load_fire16, load_smerp17
from Utils.utils import freq_tokens_per_class, split_target
from Data_Handlers.torchtext_handler import dataset2bucket_iter
from config import configuration as cfg, dataset_dir, pretrain_dir
from Logger.logger import logger


def prepare_datasets(
        train_df=None, test_df=None, stoi=None, vectors=None,
        dim=cfg['embeddings']['emb_dim'], split_test=False, get_iter=False,
        data_dir=dataset_dir, train_filename=cfg['data']['train'],
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


def clean_dataset(dataset):
    logger.info('Check if any example has no token left after cleaning.')
    for ex in dataset.examples:
        if len(ex.text) == 0:
            logger.warning(f'Examples [{ex.ids}] has no token left.')


def prepare_splitted_datasets(
        stoi=None, vectors=None, get_iter=False, dim=cfg['embeddings']['emb_dim'],
        data_dir=dataset_dir, train_dataname=cfg["data"]["train"],
        val_dataname=cfg["data"]["val"], test_dataname=cfg["data"]["test"],
        use_all_data=False, min_freq=cfg["data"]["min_freq"]):
    """ Creates train and test dataset from df and returns data loader.

    :param use_all_data: Uses all disaster data for training if True
    :param stoi:
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
    # train_df, val_df, test_df = read_csvs(data_dir=data_dir, filenames=(
    #     train_dataname, val_dataname, test_dataname))
    # logger.info(f"Train {train_df.shape}, Val {test_df.shape}, Test {val_df.shape}.")
    train_dataname = train_dataname + ".csv"
    if use_all_data:
        train_dataname = "all_training.csv"
        # if not exists(join(dataset_dir, train_dataname)):
        train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
        train_df.to_csv(join(data_dir, train_dataname))

    if stoi is None:
        logger.critical('Setting default GLOVE vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=min_freq, labelled_data=True)
    else:
        logger.critical('Setting custom vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=min_freq,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        train_vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    clean_dataset(train_dataset)

    ## Prepare labelled validation data:
    logger.info(f'Prepare labelled VALIDATION (source) data: {val_dataname}')
    # if val_df is None:
    # val_df = read_csv(data_file=val_dataname, data_dir=data_dir)
    # val_df = read_labelled_json(data_dir, val_dataname, data_set='test')
    val_dataname = val_dataname + ".csv"
    # val_df.to_csv(join(data_dir, val_dataname))
    val_dataset, (val_vocab, val_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=val_dataname, labelled_data=True)

    clean_dataset(val_dataset)

    ## Prepare labelled target data:
    logger.info(f'Prepare labelled TESTING (target) data: {test_dataname}')
    # if test_df is None:
    # test_df = read_csv(data_file=test_dataname, data_dir=data_dir)
    # test_df = read_labelled_json(data_dir, test_dataname, data_set='test')

    # if split_test:
    #     test_extra_df, test_df = split_target(df=test_df, test_size=0.4)
    test_dataname = test_dataname + ".csv"
    # test_df.to_csv(join(data_dir, test_dataname))
    test_dataset, (test_vocab, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_dataname, labelled_data=True)

    clean_dataset(test_dataset)

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


def prepare_alltrain_datasets(
        stoi=None, vectors=None, dim=cfg['embeddings']['emb_dim'],
        data_dir=dataset_dir, min_freq=cfg["data"]["min_freq"],
        all_train_dataname = "all_training.csv"):
    """ Creates a dataset by merging all the test files.

    :param all_train_dataname: Merged csv file name
    :param min_freq:
    :param stoi:
    :param vectors: Custom Vectors for each token
    :param dim: Embedding dim
    :param data_dir:
    :return:
    """
    logger.info(f'Prepare labelled TRAIN data from all Pretraining data')
    alltrain_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
    alltrain_df.to_csv(join(data_dir, all_train_dataname))
    alltrain_dataset, alltrain_vocab, alltrain_iter = prepare_single_dataset(
        stoi=stoi, vectors=vectors, dim=dim, data_dir=dataset_dir,
        min_freq=min_freq, dataname=all_train_dataname)

    return alltrain_dataset, alltrain_vocab, alltrain_iter


def prepare_single_dataset(
        stoi=None, vectors=None, dim=cfg['embeddings']['emb_dim'],
        data_dir=dataset_dir, min_freq=cfg["data"]["min_freq"],
        dataname = "all_training.csv"):
    """ Creates a torchtext dataset

    Passes a csv file for dataset creation and returns dataloader. Sets custom vectors if provided.

    :param min_freq:
    :param stoi:
    :param vectors: Custom Vectors for each token
    :param dim: Embedding dim
    :param data_dir:
    :return:
    """
    logger.info(f'Prepare labelled TRAIN data from all Pretraining data')
    if stoi is None:
        logger.critical('Setting default GLOVE vectors:')
        dataset, (vocab, label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=dataname, min_freq=min_freq, labelled_data=True)
    else:
        logger.critical('Setting custom vectors:')
        dataset, (vocab, label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=dataname, min_freq=min_freq,
            labelled_data=True, embedding_file=None, embedding_dir=None)
        vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    clean_dataset(dataset)

    logger.info('Geting train, val and test iterators')
    train_batch_size = cfg['training']['train_batch_size']
    iter = dataset2bucket_iter(dataset, batch_size=train_batch_size)

    return dataset, vocab, iter


def prepare_BERT_splitted_datasets(
        stoi=None, vectors=None, get_iter=False, dim=cfg['embeddings']['emb_dim'],
        data_dir=dataset_dir, train_dataname=cfg["data"]["train"],
        val_dataname=cfg["data"]["val"], test_dataname=cfg["data"]["test"],
        use_all_data=False, min_freq=cfg["data"]["min_freq"]):
    """ Creates train and test dataset from df and returns data loader.

    :param use_all_data: Uses all disaster data for training if True
    :param stoi:
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
    # train_df, val_df, test_df = read_csvs(data_dir=data_dir, filenames=(
    #     train_dataname, val_dataname, test_dataname))
    # logger.info(f"Train {train_df.shape}, Val {test_df.shape}, Test {val_df.shape}.")
    train_dataname = train_dataname + ".csv"
    if use_all_data:
        train_dataname = "all_training.csv"
        # if not exists(join(dataset_dir, train_dataname)):
        train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
        train_df.to_csv(join(data_dir, train_dataname))

    logger.info(f'Using BERT tokenizer')
    bert_tokenizer = BERT_tokenizer()

    if stoi is None:
        logger.critical('Setting default GLOVE vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=min_freq,
            labelled_data=True, tokenizer=bert_tokenizer.tokenize)
    else:
        logger.critical('Setting custom vectors:')
        train_dataset, (train_vocab, train_label) = get_dataset_fields(
            csv_dir=data_dir, csv_file=train_dataname, min_freq=min_freq,
            labelled_data=True, embedding_file=None, embedding_dir=None,
            tokenizer=bert_tokenizer.tokenize)
        train_vocab.vocab.set_vectors(stoi=stoi, vectors=vectors, dim=dim)

    clean_dataset(train_dataset)

    ## Prepare labelled validation data:
    logger.info(f'Prepare labelled VALIDATION (source) data: {val_dataname}')
    # if val_df is None:
    # val_df = read_csv(data_file=val_dataname, data_dir=data_dir)
    # val_df = read_labelled_json(data_dir, val_dataname, data_set='test')
    val_dataname = val_dataname + ".csv"
    # val_df.to_csv(join(data_dir, val_dataname))
    val_dataset, (val_vocab, val_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=val_dataname, labelled_data=True)

    clean_dataset(val_dataset)

    ## Prepare labelled target data:
    logger.info(f'Prepare labelled TESTING (target) data: {test_dataname}')
    # if test_df is None:
    # test_df = read_csv(data_file=test_dataname, data_dir=data_dir)
    # test_df = read_labelled_json(data_dir, test_dataname, data_set='test')

    # if split_test:
    #     test_extra_df, test_df = split_target(df=test_df, test_size=0.4)
    test_dataname = test_dataname + ".csv"
    # test_df.to_csv(join(data_dir, test_dataname))
    test_dataset, (test_vocab, test_label) = get_dataset_fields(
        csv_dir=data_dir, csv_file=test_dataname, labelled_data=True)

    clean_dataset(test_dataset)

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


def create_unlabeled_datasets(
        s_lab_df=None, data_dir: str = dataset_dir, labelled_source_name: str
        = cfg['data']['train'],
        unlabelled_source_name: str = None, unlabelled_target_name: str = None):
    """ creates vocab and other info.

    :param s_lab_df:
    :param data_dir:
    :param labelled_source_name:
    :param unlabelled_source_name:
    :param unlabelled_target_name:
    :return:
    """
    logger.info('creates vocab and other info.')

    if unlabelled_source_name is None:
        unlabelled_source_name = cfg["data"]["source"]['unlabelled'],
        # labelled_target_name=cfg['data']['test'],
        unlabelled_target_name = cfg["data"]["target"]['unlabelled']
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
