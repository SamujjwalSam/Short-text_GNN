# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Code to handle text datasets using TorchText
__description__ : Code to handle text datasets using TorchText
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
import pickle
import dill
from pathlib import Path
from os.path import join
from torchtext import data

from config import configuration as cfg, device
from Logger.logger import logger


def prepare_fields(text_headers: list = None, label_headers=None, tokenizer='spacy',
                   batch_first: bool = True, include_lengths: bool = True,
                   n_classes: int = cfg['data']['num_classes']):
    """ Generates fields present on the dataset.

    Args:
        text_headers:
        label_headers:
        tokenizer:
        batch_first:
        include_lengths:

    Returns:
    :param include_lengths:
    :param batch_first:
    :param tokenizer:
    :param label_headers:
    :param text_headers:
    :param n_classes:

    """
    ## Define field types:
    if text_headers is None:
        text_headers = ['text']
    if label_headers is None:
        label_headers = [str(cls) for cls in range(n_classes)]

    TEXT = data.Field(tokenize=tokenizer, batch_first=batch_first,
                      include_lengths=include_lengths)
    LABEL = data.LabelField(dtype=torch.float, batch_first=batch_first,
                            use_vocab=False, sequential=False)
    IDS = data.LabelField(batch_first=batch_first, use_vocab=False,
                          sequential=False)
    # IDS = data.Field(batch_first=batch_first, use_vocab=False,
    #                  sequential=False)

    # labelled_fields = [("id", None)]
    labelled_fields = [("ids", IDS)]
    # unlabelled_fields = [("id", None)]
    unlabelled_fields = [("ids", IDS)]

    for header in text_headers:
        labelled_fields.append((header, TEXT))
        unlabelled_fields.append((header, TEXT))

    for header in label_headers:
        labelled_fields.append((header, LABEL))

    # unlabelled_fields = [("id", IDS)]
    # # unlabelled_fields = [("id", None)]
    # for header in text_headers:
    #     unlabelled_fields.append((header, TEXT))

    return (TEXT, LABEL), labelled_fields, unlabelled_fields


def create_dataset(examples, fields=None):
    """ Creates a TorchText Dataset from examples (list) and fields (dict).

    :param fields:
    :param examples:

    """
    if fields is None:
        _, fields, unlabelled_fields = prepare_fields()

    dataset = data.Dataset(examples=examples, fields=fields)

    logger.debug(vars(dataset.examples[0]))
    return dataset


def create_tabular_dataset(csv_file: str, data_dir: str, fields=None,
                           skip_header: bool = True) -> data.dataset.TabularDataset:
    """ Reads a csv file and returns TorchText TabularDataset format.

    Args:
        csv_file:
        fields:
        skip_header:

    Returns:

    """
    if fields is None:
        _, fields, unlabelled_fields = prepare_fields()

    dataset = data.TabularDataset(
        path=join(data_dir, csv_file), format='csv', fields=fields,
        skip_header=skip_header)

    logger.debug(vars(dataset.examples[0]))
    return dataset


def split_dataset(dataset, split_size=0.7, stratify=False, strata_name='label'):
    """ Splits a torchtext dataset into train, test, val.

    :param dataset:
    :param split_size:
    :param stratify:
    :param strata_name:
    :return:
    """
    train, test = dataset.split(
        split_ratio=split_size, stratified=stratify, strata_field=strata_name)
    return train, test


def save_to_pickle(dataSetObject, PATH):
    with open(PATH, 'wb') as output:
        for i in dataSetObject:
            pickle.dump(vars(i), output, pickle.HIGHEST_PROTOCOL)


def load_pickle(PATH, FIELDNAMES, FIELD):
    dataList = []
    with open(PATH, "rb") as input_file:
        while True:
            try:
                # Taking the dictionary instance as the input Instance
                inputInstance = pickle.load(input_file)
                # plugging it into the list
                dataInstance = [inputInstance[FIELDNAMES[0]], inputInstance[FIELDNAMES[1]]]
                # Finally creating an example objects list
                dataList.append(data.Example().fromlist(dataInstance, fields=FIELD))
            except EOFError:
                break

    # At last creating a data Set Object
    exampleListObject = data.Dataset(dataList, fields=FIELD)
    return exampleListObject


def save_dataset(dataset, save_dir, name, fields=None):
    ename = name + "_examples.pkl"
    fname = name + "_fields.pkl"
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
        ename = Path(ename)
        fname = Path(fname)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, save_dir / ename, pickle_module=dill)
    if fields:
        torch.save(fields, save_dir / fname, pickle_module=dill)


def load_dataset(load_dir, name):
    ename = name + "_examples.pkl"
    fname = name + "_fields.pkl"
    if not isinstance(load_dir, Path):
        load_dir = Path(load_dir)
        ename = Path(ename)
        fname = Path(fname)
    dataset = torch.load(load_dir / ename, pickle_module=dill)
    try:
        fields = torch.load(load_dir / fname, pickle_module=dill)
        return data.Dataset(dataset, fields)
    except FileNotFoundError:
        logger.warning(f'fields not found at: '
                       f'[{load_dir / fname}]')
    return dataset


def create_vocab(dataset: data.dataset.TabularDataset, TEXT_field: data.field.Field,
                 LABEL_field: data.field.LabelField = None, embedding_file: [None, str] = None,
                 embedding_dir: [None, str] = None, min_freq: int = 2,
                 show_vocab_details: bool = True) -> None:
    """ Creates vocabulary using TorchText.

    :param dataset:
    :param TEXT_field:
    :param LABEL_field:
    :param embedding_file:
    :param embedding_dir:
    :param min_freq:
    :param show_vocab_details:
    """
    if embedding_file is not None:
        # initialize embeddings (Glove)
        logger.info(f'Using embedding file [{embedding_file}]')
        TEXT_field.build_vocab(
            dataset, min_freq=min_freq, vectors=embedding_file,
            vectors_cache=embedding_dir)
    else:
        TEXT_field.build_vocab(dataset, min_freq=min_freq)

    if LABEL_field:
        LABEL_field.build_vocab(dataset)
        # No. of unique label types
        logger.info(f"Size of LABEL vocabulary: {len(LABEL_field.vocab)}")

    if show_vocab_details:
        # No. of unique tokens in text
        logger.info(f"Size of TEXT vocabulary: {len(TEXT_field.vocab)}")

        # Commonly used tokens
        logger.info("10 most common tokens in vocabulary: {}".format(
            TEXT_field.vocab.freqs.most_common(10)))


def dataset2iter(datasets: tuple, batch_size=None, batch_sizes=(128, 256, 256),
                 shuffle=True):
    """
    Converts dataset (DataFrame) to TorchText iterator.

    Returns:

    """
    # data_df.to_csv(save_path, header=headers)

    # datasets = create_tabular_dataset(save_path, fields)

    if batch_size:
        iterator = data.Iterator.splits(
            datasets, batch_size=batch_size, shuffle=shuffle, sort=False,
            repeat=False, device=device)
        # batch_sizes=batch_sizes,
        # sort_key=lambda x: len(x.text),
        # sort_within_batch=True,

    else:
        iterator = data.Iterator.splits(
            datasets,
            # batch_size=batch_size,
            batch_sizes=batch_sizes,
            shuffle=shuffle,
            sort=False,
            repeat=False,
            # sort_key=lambda x: len(x.text),
            # sort_within_batch=True,
            device=device)
    return iterator


def dataset2bucket_iter(datasets: tuple, batch_size=None, batch_sizes: tuple = (32, 64, 64)):
    """ Converts dataset (DataFrame) to TorchText iterator.

    :param datasets:
    :param batch_size:
    :param batch_sizes:
    :return:
    """
    # data_df.to_csv(save_path, header=headers)

    # datasets = create_tabular_dataset(save_path, fields)

    if batch_size is not None:
        iterator = data.BucketIterator(
            datasets,
            batch_size=batch_size,
            # batch_sizes=batch_sizes,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=True,
            device=device)
    else:
        iterator = data.BucketIterator.splits(
            datasets,
            # batch_size=batch_size,
            batch_sizes=batch_sizes,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            shuffle=True,
            device=device)
    return iterator


class MultiIterator:
    """https://github.com/pytorch/text/issues/375"""

    def __init__(self, iter_list):
        """MultiIterator to chain multiple iterators into one.

        Parameters
        ----------
        iter_list : [list]
            Sequence of torchtext iterators.
        """
        self.iters = iter_list

    def __iter__(self):
        for it in self.iters:
            for batch in it:
                yield batch

    def __len__(self):
        return sum(len(it) for it in self.iters)


def torchtext_batch2multilabel(batch, label_cols=None, n_classes=cfg['data']['num_classes']):
    """ Returns labels for a TorchText batch.

    Args:
        batch:
        label_cols:

    Returns:

    """
    if label_cols is None:
        label_cols = [str(cls) for cls in range(n_classes)]
    return torch.cat([getattr(batch, feat).unsqueeze(1) for feat in label_cols],
                     dim=1).float()
