# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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
import timeit
import argparse
import numpy as np
import pandas as pd
from os.path import join
from json import dumps, dump

from simpletransformers.classification import MultiLabelClassificationModel

from config import configuration as cfg, platform as plat, username as user
from Utils.utils import calculate_performance
from Logger.logger import logger


def format_inputs(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer.

    """
    tf_df = pd.DataFrame(df[df.columns[:1]], columns=['text'])
    labels = []
    for idx, row in df[df.columns[1:]].iterrows():
        labels.append(row.to_list())
    tf_df['labels'] = pd.Series(labels, index=tf_df.index)
    return tf_df


def macro_f1(labels, preds, threshold=0.5):
    """ Converts probabilities to labels using the [threshold] and calculates
    metrics.

    Parameters
    ----------
    labels
    preds
    label_names
    threshold

    Returns
    -------

    """
    np.savetxt(join(cfg['paths']["dataset_dir"][plat][user],
                    cfg['data']["dataset_name"] + "_" +
                    cfg['model']['model_type'] + '_labels.txt'),
               labels)
    np.savetxt(join(cfg['paths']["dataset_dir"][plat][user],
                    cfg['data']["dataset_name"] + "_" +
                    cfg['model']['model_type'] + '_preds.txt'),
               preds)

    logger.info("labels:\n[{}]".format(labels))
    logger.info("preds:\n[{}]".format(preds))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    logger.info("preds with threshold [{}]:\n[{}]".format(threshold, preds))

    scores = calculate_performance(labels, preds)

    return scores


def main(train_df, test_df, n_classes=4,
         dataset_name=cfg["data"]["source"]['labelled'],
         model_name=cfg['transformer']['model_name'],
         model_type=cfg['transformer']['model_type'],
         num_epoch=cfg['sampling']['num_train_epoch'],
         use_cuda=cfg['model']['use_cuda']):
    """Train and Evaluation data needs to be in a Pandas Dataframe containing at
    least two columns, a 'text' and a 'labels' column. The `labels` column
    should contain multi-hot encoded lists.

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
    train_df = format_inputs(train_df)
    test_df = format_inputs(test_df)

    ## Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        model_name, model_type, num_labels=n_classes,
        args={
            'output_dir':                     cfg['paths']['result_dir'],
            'cache_dir':                      cfg['paths']['cache_dir'],
            'fp16':                           False,
            'fp16_opt_level':                 'O1',
            'max_seq_length':                 cfg['model']['max_seq_len'],
            'weight_decay':                   cfg['transformer'][
                                                  'weight_decay'],
            'learning_rate':                  cfg['transformer'][
                                                  'learning_rate'],
            'adam_epsilon':                   cfg['transformer'][
                                                  'adam_epsilon'],
            'warmup_ratio':                   cfg['transformer'][
                                                  'warmup_ratio'],
            'warmup_steps':                   cfg['transformer'][
                                                  'warmup_steps'],
            'max_grad_norm':                  cfg['transformer'][
                                                  'max_grad_norm'],
            'train_batch_size':               cfg['sampling'][
                                                  'train_batch_size'],
            'gradient_accumulation_steps':    cfg['model'][
                                                  'gradient_accumulation_steps'],
            'eval_batch_size':                cfg['sampling'][
                                                  'eval_batch_size'],
            'num_train_epochs':               num_epoch,
            'evaluate_during_training':       False,
            'evaluate_during_training_steps': 3000,
            'save_steps':                     10000,
            'overwrite_output_dir':           True,
            'reprocess_input_data':           True,
            'n_gpu':                          1,
            'threshold':                      0.5
        }, use_cuda=use_cuda)

    ## Train the model
    start_time = timeit.default_timer()
    model.train_model(train_df, eval_df=test_df)
    train_time = timeit.default_timer() - start_time

    ## Evaluate the model
    start_time = timeit.default_timer()
    result, model_outputs, wrong_predictions = model.eval_model(
        test_df, macro_f1=macro_f1)
    prediction_time = timeit.default_timer() - start_time

    ## Analyze wrong predictions
    logger.info("Wrong prediction count: [{}]".format(len(wrong_predictions)))
    logger.info("Wrong predictions: ")
    miss_ids = []
    miss_texts = []
    # misses = {"ids": miss_ids,"texts":miss_texts}
    for example in wrong_predictions:
        logger.info("Misclassification sample id: [{}], text: [{}]".format(
            example.guid, example.text_a))
        miss_ids.append(example.guid)
        miss_texts.append(example.text_a)

    missed_samples = pd.DataFrame({"ids": miss_ids, "texts": miss_texts})
    missed_file = dataset_name + model_type + num_epoch + "missed_samples.csv"
    logger.info("Saving wrongly classified samples in file: [{}]"
                .format(missed_file))
    missed_samples.to_csv(missed_file)

    logger.info(dumps(result, indent=4))
    with open(join(cfg['paths']["dataset_dir"][plat][user],
                   cfg['data']["dataset_name"] + "_" + model_name +
                   '_result.json'), 'w') as f:
        dump(result, f)

    logger.info("Total training time for [{}] with [{}] samples: [{} sec]"
                "\nPer sample: [{} sec] for model: [{}]"
                .format(num_epoch, train_df.shape[0], train_time, train_time /
                        train_df.shape[0], model_type))

    logger.info("Total prediction time for [{}] samples: [{} sec]"
                "\nPer sample: [{} sec] for model: [{}]"
                .format(test_df.shape[0], prediction_time, prediction_time /
                        test_df.shape[0], model_type))

    ## For prediction just pass a list of texts.
    # predictions, raw_outputs = model.predict(
    #     ['This thing is entirely different from the other thing. '])
    # predictions, raw_outputs = model.predict(test_df)
    # logger.debug(predictions)
    # logger.debug(raw_outputs)

    return result, model_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-d", "--dataset_name",
                        default=cfg['data']['dataset_name'], type=str)
    parser.add_argument("-m", "--model_name",
                        default=cfg['model']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type",
                        default=cfg['model']['model_type'], type=str)
    parser.add_argument("-ne", "--num_train_epochs",
                        default=cfg['sampling']['num_train_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['model']['use_cuda'], action='store_true')

    args = parser.parse_args()

    main(args.dataset_name, args.model_name, args.model_type,
         args.num_train_epochs, args.use_cuda)
