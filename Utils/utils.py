# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Short summary of the script.
__description__ : Details and usage.
__project__     : Tweet_Classification
__classes__     : utils
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
import pandas as pd
import numpy as np
import scipy.sparse as sp
from functools import partial
from os.path import join
from collections import OrderedDict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.measures import\
    get_combination_wise_output_matrix

from config import configuration as cfg, dataset_dir
from File_Handlers.json_handler import read_labelled_json
from Text_Processesor.tweet_normalizer import normalizeTweet
from Logger.logger import logger


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ dot product between 2 tensors for dense and sparse format.

    :param x:
    :param y:
    :return:
    """
    if x.is_sparse:
        res = torch.spmm(x, y)
    else:
        res = torch.matmul(x, y)
    return res


def iterative_train_test_split(X, y, test_size, order=2, random_state=None):
    """Iteratively stratified train/test split

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
        :param order:
        :param random_state:
    """

    stratifier = IterativeStratification(n_splits=2, order=order,
                                         sample_distribution_per_fold=[test_size, 1.0 - test_size],
                                         random_state=random_state)
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]

    return X_train, y_train, X_test, y_test


def split_target(df=None, data_dir=dataset_dir, labelled_dataname=cfg['data']['test'],
                 test_size=0.999, train_size=None, n_classes=cfg['data']['num_classes'], stratified=False):
    """ Splits labelled target data to train and test set.

    :param data_dir:
    :param labelled_dataname:
    :param test_size:
    :param train_size:
    :param n_classes:
    :return:
    """
    logger.info('Splits labelled target data to train and test set.')
    ## Read target data
    if df is None:
        df = read_labelled_json(data_dir, labelled_dataname)
    df, t_lab_test_df = split_df(df, test_size=test_size, stratified=stratified,
                                 order=2, n_classes=n_classes)

    logger.info(f'Number of TEST samples: [{t_lab_test_df.shape[0]}]')

    if train_size is not None:
        _, df = split_df(df, test_size=train_size,
                         stratified=stratified, order=2, n_classes=n_classes)
    logger.info(f'Number of TRAIN samples: [{df.shape[0]}]')

    # token_dist(t_lab_df)

    return df, t_lab_test_df


def sp_coo2torch_coo(M: sp.csr.csr_matrix) -> torch.sparse:
    """

    :param M:
    :return:
    """
    if isinstance(M, sp.csr_matrix):
        M = M.tocoo()

    M = torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((M.row, M.col))),
        torch.FloatTensor(M.data),
        torch.Size(M.shape))

    return M


def logit2label(predictions_df: pd.core.frame.DataFrame, cls_thresh: [list, float],
                drop_irrelevant=False, return_df=False):
    """ Converts logit to multi-hot based on threshold per class.

    :param predictions_df: can be pd.DataFrame or np.NDArray or torch.tensor
    :param cls_thresh: List of floats as threshold for each class
    :param drop_irrelevant: Remove samples for which no class crossed it's
    threshold. i.e. [0.,0.,0.,0.]
    """
    if isinstance(predictions_df, pd.core.frame.DataFrame):
        logger.debug((predictions_df.values.min(), predictions_df.values.max()))
        df_np = predictions_df.to_numpy()
    elif isinstance(predictions_df, (np.ndarray, torch.Tensor)):
        df_np = predictions_df
    else:
        NotImplementedError(f'Only supports pd.DataFrame or np.ndarray or '
                            f'torch.Tensor but received [{type(predictions_df)}]')

    ## Create threshold list for all classes if only one threshold float is provided:
    if isinstance(cls_thresh, float):
        cls_thresh = [cls_thresh for i in range(df_np.shape[1])]

    for col in range(df_np.shape[1]):
        df_np[:, col][df_np[:, col] > cls_thresh[col]] = 1.
        df_np[:, col][df_np[:, col] <= cls_thresh[col]] = 0.

    if return_df:
        predictions_df = pd.DataFrame(df_np, index=predictions_df.index)

        if drop_irrelevant:
            # delete all rows where sum == 0
            irrelevant_rows = []
            for i, row in predictions_df.iterrows():
                if sum(row) < 1:
                    irrelevant_rows.append(i)

            predictions_df = predictions_df.drop(irrelevant_rows)
        return predictions_df
    else:
        return df_np


# No. of trianable parameters
def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'The model has {param_count:,} trainable parameters')
    return param_count


def split_data(lab_tweets, test_size=0.3, stratified=True, random_state=0,
               order=2, n_classes=cfg['data']['num_classes']):
    """ Splits json data.

    :param lab_tweets:
    :param test_size:
    :param stratified:
    :param random_state:
    :return:
    """
    if isinstance(lab_tweets, dict):
        sample_keys = np.array(list(lab_tweets.keys()))
    else:  ## For DataFrame
        sample_keys = lab_tweets.index.to_array()

    if stratified is False:
        train_split, test_split = train_test_split(sample_keys,
                                                   test_size=test_size,
                                                   random_state=random_state)
    else:
        if isinstance(lab_tweets, dict):
            classes = []
            for idx, val in lab_tweets.items():
                classes.append(val["classes"])

            mlb = MultiLabelBinarizer()
            train_y_hot = mlb.fit_transform(classes)
        else:  ## For DataFrame
            train_y_hot = lab_tweets[lab_tweets.columns[-n_classes:]].to_array()

        sample_keys = np.reshape(sample_keys, (sample_keys.shape[0], -1))

        logger.info(Counter(
            combination for row in get_combination_wise_output_matrix(
                train_y_hot, order=order) for combination in row))

        train_split, y_train, test_split, y_test = iterative_train_test_split(
            sample_keys, train_y_hot, test_size=test_size)

        logger.info(Counter(combination for row in
                            get_combination_wise_output_matrix(y_train,
                                                               order=2)
                            for combination in row))

        logger.info(Counter(combination for row in
                            get_combination_wise_output_matrix(y_test,
                                                               order=2)
                            for combination in row))

        train_split = np.reshape(train_split, (train_split.shape[0],))
        test_split = np.reshape(test_split, (test_split.shape[0],))

    train = OrderedDict()
    test = OrderedDict()
    for id in train_split:
        train[id] = lab_tweets[id]
    for id in test_split:
        test[id] = lab_tweets[id]
    return train, test


def split_df(df, test_size=0.3, stratified=True, random_state=0,
             order=2, n_classes=cfg['data']['num_classes']):
    """ Splits Dataframe.

    :param n_classes:
    :param order:
    :param df:
    :param test_size:
    :param stratified:
    :param random_state:
    :return:
    """
    sample_keys = df.index.values

    if stratified is False:
        train_split, test_split = train_test_split(sample_keys,
                                                   test_size=test_size,
                                                   random_state=random_state)
        train = df.loc[train_split]
        test = df.loc[test_split]
    else:
        train_y_hot = df[df.columns[-n_classes:]].values

        sample_keys = np.reshape(sample_keys, (sample_keys.shape[0], -1))

        logger.info(Counter(
            combination for row in get_combination_wise_output_matrix(
                train_y_hot, order=order) for combination in row))

        train_split, y_train, test_split, y_test = iterative_train_test_split(
            sample_keys, train_y_hot, test_size=test_size)

        logger.info('Training Class distribution: [{}]'.format(Counter(
            combination for row in get_combination_wise_output_matrix(
                y_train, order=2) for combination in row)))

        logger.info('Testing Class distribution: [{}]'.format(Counter(
            combination for row in get_combination_wise_output_matrix(
                y_test, order=2) for combination in row)))

        train_split = np.reshape(train_split, (train_split.shape[0],))
        test_split = np.reshape(test_split, (test_split.shape[0],))

        train = df.loc[train_split]
        test = df.loc[test_split]

    return train, test


tokenizer = partial(normalizeTweet, return_tokens=True)


def token_dist(df, txt_tokenized=False):
    """ Returns a dict of most freq tokens per class.

    :param df: Labelled DataFrame ['text', label_1, label_2, ...]
    :param txt_tokenized:
    :return:
    """
    cls_freq = {}
    all_tokens = []
    for col in df.columns[1:]:
        col_series = df.loc[df[col] == 1][df.columns[0]]
        if not txt_tokenized:
            col_series = col_series.apply(tokenizer)

        for tokens in col_series.to_list():
            all_tokens += tokens

        cls_freq[col] = Counter(all_tokens)

    return cls_freq, set(all_tokens)


def freq_tokens_per_class(df: pd.core.frame.DataFrame, normalize: bool = True):
    """ Returns a dict of most freq tokens per class.

    :param df: Labelled DataFrame ['text', label_1, label_2, ...]
    :param normalize: If values devided by total number of samples of that class
    :return:
    """
    token_cls_freq = {}
    for i, tweet in df.iterrows():
        tweet_toks = set(tokenizer(tweet.text))
        for token in tweet_toks:
            # token_cls_freq[token] = {}
            num_classes = len(tweet[1:])
            try:
                token_cls_freq[token]
            except KeyError:
                token_cls_freq[token] = [0.0] * num_classes
            for cls, (_, val) in zip(range(num_classes), tweet[1:].items()):
                if val == 1:
                    token_cls_freq[token][int(cls)] += 1
            # for cls, val in tweet[1:].items():
            #     if val == 1:
            #         token_cls_freq[token][int(cls)] += 1
            # try:
            #     token_cls_freq[token][cls] += 1
            # except KeyError:
            #     token_cls_freq[token][cls] = 1

    if normalize:
        cls_counts = []
        for cls in df.columns[1:]:
            cls_counts.append(sum(df[cls]))
        for token_id, cls_list in token_cls_freq.items():
            token_cls_freq[token_id] = np.divide(cls_list, cls_counts).tolist()

    return token_cls_freq


def token_dist2token_labels(cls_freq: dict, vocab_set: list):
    token_labels = {}
    for token in vocab_set:
        l_vec = [0] * len(cls_freq)
        for cls in cls_freq:
            freq = cls_freq[cls][token]
            l_vec[cls] = freq
        token_labels[token] = l_vec

    return token_labels


def merge_dicts(*dict_args):
    """ Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts. """
    result = OrderedDict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def save_model(model, saved_model_name='tweet_bilstm', saved_model_dir=dataset_dir):
    try:
        torch.save(model.state_dict(), join(saved_model_dir, saved_model_name))
    except Exception as e:
        logger.fatal(
            "Could not save model at [{}] due to Error: [{}]".format(join(
                saved_model_dir, saved_model_name), e))
        return False
    return True


def load_model(model, saved_model_name='tweet_bilstm', saved_model_dir=dataset_dir):
    try:
        model.load_state_dict(
            torch.load(join(saved_model_dir, saved_model_name)))
    except Exception as e:
        logger.fatal(
            "Could not load model from [{}] due to Error: [{}]".format(join(
                saved_model_dir, saved_model_name), e))
        return False
    model.eval()
    return model


from networkx.readwrite.gpickle import write_gpickle, read_gpickle


def save_graph(G, graph_path='graph.pkl'):
    # save graphs and labels
    write_gpickle(G, graph_path)
    logger.info(f'Saved graph at [{graph_path}]')


def load_graph(graph_path='graph.pkl'):
    # load processed data from directory graph_path
    logger.info(f'Loading graph from [{graph_path}]')
    G = read_gpickle(graph_path)
    return G


def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    pass


if __name__ == "__main__":
    main()
