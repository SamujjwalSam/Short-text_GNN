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
from os.path import join
from torchtext import data

from Logger.logger import logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_fields(text_headers=None, label_headers=None, tokenizer='spacy',
                   batch_first=True, include_lengths=True, n_classes=7):
    """ Generates fields present on the dataset.

    Args:
        text_headers:
        label_headers:
        tokenizer:
        batch_first:
        include_lengths:

    Returns:

    """
    ## Define field types:
    if text_headers is None:
        text_headers = ['text']
    if label_headers is None:
        # label_headers = ['0', '1', '2', '3', '4', '5', '6']
        label_headers = [str(cls) for cls in range(n_classes)]

    TEXT = data.Field(tokenize=tokenizer, batch_first=batch_first,
                      include_lengths=include_lengths)
    LABEL = data.LabelField(dtype=torch.float, batch_first=batch_first,
                            use_vocab=False, sequential=False)
    IDS = data.LabelField(batch_first=batch_first, use_vocab=False,
                          sequential=False)

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


def create_tabular_dataset(csv_file, data_dir, fields=None, skip_header=True):
    """ Reads a csv file and returns TorchText TabularDataset format.

    Args:
        csv_file:
        fields:
        skip_header:

    Returns:

    """
    if fields is None:
        _, fields, unlabelled_fields = prepare_fields()

    data_table = data.TabularDataset(path=join(data_dir, csv_file),
                                     format='csv', fields=fields,
                                     skip_header=skip_header)

    logger.debug(vars(data_table.examples[0]))

    return data_table


def create_vocab(dataset, TEXT_field, LABEL_field=None, embedding_file=None,
                 embedding_dir=None, min_freq=2, show_vocab_details=True):
    if embedding_file is not None:
        # initialize embeddings (Glove)
        TEXT_field.build_vocab(dataset, min_freq=min_freq,
                               vectors=embedding_file,
                               vectors_cache=embedding_dir)
    else:
        TEXT_field.build_vocab(dataset, min_freq=min_freq)

    if LABEL_field:
        LABEL_field.build_vocab(dataset)
        # No. of unique tokens in label
        logger.info("Size of LABEL vocabulary: {}".format(len(
            LABEL_field.vocab)))

    if show_vocab_details:
        # No. of unique tokens in text
        logger.info("Size of TEXT vocabulary: {}".format(len(TEXT_field.vocab)))

        # Commonly used tokens
        logger.info("10 most common tokens in vocabulary: {}".format(
            TEXT_field.vocab.freqs.most_common(10)))


def df2iter(datatable, batch_size=None, batch_sizes=(32, 64, 64),):
    """
    Converts DataFrame to TorchText iterator.

    Returns:

    """
    # data_df.to_csv(save_path, header=headers)

    # datatable = create_tabular_dataset(save_path, fields)

    if batch_size:
        iterator = data.BucketIterator.splits(
            (datatable),
            batch_size=batch_size,
            # batch_sizes=batch_sizes,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device)
    else:
        iterator = data.BucketIterator.splits(
            (datatable),
            # batch_size=batch_size,
            batch_sizes=batch_sizes,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
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


def torchtext_batch2multilabel(batch, label_cols=None, n_classes=7):
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
