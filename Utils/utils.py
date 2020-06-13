# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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
from json import load, loads
from os.path import join, exists
from collections import OrderedDict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, recall_score, precision_score,\
    f1_score, precision_recall_fscore_support, classification_report
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import\
    get_combination_wise_output_matrix

from config import configuration as cfg, platform as plat, username as user
from Logger.logger import logger


def split_data(lab_tweets, split_size=0.3, stratified=True, random_state=0):
    """ Splits json data.

    :param lab_tweets:
    :param split_size:
    :param stratified:
    :param random_state:
    :return:
    """
    sample_keys = np.array(list(lab_tweets.keys()))

    if stratified is False:
        train_split, test_split = train_test_split(sample_keys,
                                                   test_size=split_size,
                                                   random_state=random_state)
    else:
        classes = []
        for idx, val in lab_tweets.items():
            classes.append(val["classes"])

        mlb = MultiLabelBinarizer()
        train_y_hot = mlb.fit_transform(classes)

        sample_keys = np.reshape(sample_keys, (sample_keys.shape[0], -1))

        logger.info(Counter(combination for row in
                            get_combination_wise_output_matrix(train_y_hot,
                                                               order=2)
                            for combination in row))

        train_split, y_train, test_split, y_test = iterative_train_test_split(
            sample_keys, train_y_hot, test_size=split_size)

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


def save_model(model, saved_model_name='tweet_bilstm',
               saved_model_dir=cfg["paths"]["dataset_dir"][plat][user]):
    try:
        torch.save(model.state_dict(), join(saved_model_dir, saved_model_name))
    except Exception as e:
        logger.fatal(
            "Could not save model at [{}] due to Error: [{}]".format(join(
                saved_model_dir, saved_model_name), e))
        return False
    return True


def load_model(model, saved_model_name='tweet_bilstm',
               saved_model_dir=cfg["paths"]["dataset_dir"][plat][user]):
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


def load_json(filename: str, filepath: str = '', ext: str = ".json",
              show_path: bool = True) -> OrderedDict:
    """ Reads json file as a Python OrderedDict.

    :param filename:
    :param filepath:
    :param ext:
    :param show_path:
    :return:
    """
    file_loc = join(filepath, filename + ext)
    if show_path:
        logger.debug("Reading JSON file: [{}]".format(file_loc))
    if exists(file_loc):
        try:
            with open(file_loc, encoding="utf-8") as file:
                json_dict = load(file)
                json_dict = OrderedDict(json_dict)
            file.close()
            return json_dict
        except Exception as e:
            logger.debug(
                "Could not open file as JSON: [{}]. \n Reason:[{}]".format(
                    file_loc, e))

            logger.warn("Reading JSON as STR: [{}]".format(file_loc))
            with open(file_loc, encoding="utf-8") as file:
                json_dict = str(file)
                json_dict = loads(json_dict)
            return json_dict
    else:
        raise FileNotFoundError("File not found at: [{}]".format(file_loc))
        # logger.error("File does not exist at: [{}]".format(file_loc))
        # return False


def json_keys2df(data_keys, json_data=None, json_filename=None,
                 dataset_dir: str= '', replace_index=False) -> \
        pd.core.frame.DataFrame:
    """ Converts json tweet data to a dataframe for only selected data_keys.

    :param json_filename:
    :param data_keys: list of str to be read from the json and made a column
    in df.
    :param json_data:
    :param dataset_dir:
    :return:
    """
    if json_data is None:
        json_data = load_json(filename=json_filename, filepath=dataset_dir)
    #     logger.debug(json_data)
    json_df = pd.DataFrame.from_dict(json_data, orient='index', dtype=object,
                                     columns=data_keys)
    if replace_index:  ## Replaces tweet ids by autoincremented int
        json_df.index = pd.Series(range(0, json_df.shape[0]))

    return json_df


def logit2label(predictions_df: pd.core.frame.DataFrame, cls_thresh: list,
                drop_irrelevant=True):
    """ Converts logit to multi-hot based on threshold per class.

    :param predictions_df:
    :param cls_thresh:
    :param drop_irrelevant: Remove samples for which no class crossed it's
    threshold. i.e. [0,0,0,0]
    """
    logger.debug((predictions_df.values.min(), predictions_df.values.max()))
    df_np = predictions_df.to_numpy()
    for col in range(df_np.shape[1]):
        df_np[:, col][df_np[:, col] > cls_thresh[col]] = 1.
        df_np[:, col][df_np[:, col] <= cls_thresh[col]] = 0.

    predictions_df = pd.DataFrame(df_np, index=predictions_df.index)

    if drop_irrelevant:
        # delete all rows where sum == 0
        irrelevant_rows = []
        for i, row in predictions_df.iterrows():
            if sum(row) < 1:
                irrelevant_rows.append(i)

        predictions_df = predictions_df.drop(irrelevant_rows)
    return predictions_df


def calculate_performance(true: np.ndarray, pred: np.ndarray) -> dict:
    """

    Parameters
    ----------
    true: Multi-hot
    pred: Multi-hot

    Returns
    -------

    """
    scores = {"accuracy": {}}
    scores["accuracy"]["unnormalize"] = accuracy_score(true, pred)
    scores["accuracy"]["normalize"] = accuracy_score(true, pred, normalize=True)

    scores["precision"] = {}
    scores["precision"]["classes"] = precision_score(true, pred,
                                                     average=None).tolist()
    scores["precision"]["weighted"] = precision_score(true, pred,
                                                      average='weighted')
    scores["precision"]["micro"] = precision_score(true, pred, average='micro')
    scores["precision"]["macro"] = precision_score(true, pred, average='macro')
    scores["precision"]["samples"] = precision_score(true, pred,
                                                     average='samples')

    scores["recall"] = {}
    scores["recall"]["classes"] = recall_score(true, pred,
                                               average=None).tolist()
    scores["recall"]["weighted"] = recall_score(true, pred, average='weighted')
    scores["recall"]["micro"] = recall_score(true, pred, average='micro')
    scores["recall"]["macro"] = recall_score(true, pred, average='macro')
    scores["recall"]["samples"] = recall_score(true, pred, average='samples')

    scores["f1"] = {}
    scores["f1"]["classes"] = f1_score(true, pred, average=None).tolist()
    scores["f1"]["weighted"] = f1_score(true, pred, average='weighted')
    scores["f1"]["micro"] = f1_score(true, pred, average='micro')
    scores["f1"]["macro"] = f1_score(true, pred, average='macro')
    scores["f1"]["samples"] = f1_score(true, pred, average='samples')

    return scores


def flatten_results(results: dict):
    """ Flattens the nested result dict and save as csv.

    :param results:
    :return:
    """
    ## Replace classes list to dict:
    for i, result in enumerate(results):
        for approach, vals in result.items():
            if approach != 'params':
                for metric1, averaging in vals.items():
                    for avg, score in averaging.items():
                        if avg == 'classes':
                            classes_dict = {}
                            for cls, val in enumerate(score):
                                cls = str(cls)
                                classes_dict[cls] = val
                            results[i][approach][metric1][avg] = classes_dict

    result_df = pd.json_normalize(results, sep='_')

    ## Round values and save:
    # result_df.round(decimals=4).to_csv('results.csv')

    return result_df


# No. of trianable parameters
def count_parameters(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'The model has {param_count:,} trainable parameters')
    return param_count


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
