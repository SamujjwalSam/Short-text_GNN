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

import torch
import timeit
import argparse
import numpy as np
import pandas as pd
from os.path import join
# from json import dumps, dump
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Metrics.metrics import calculate_performance_pl, calculate_performance_sk
from Logger.logger import logger


def format_inputs(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer.

    """
    df['labels'] = df[df.columns[1:]].values.tolist()
    df = df[['text', 'labels']].copy()
    return df


def macro_f1(labels, preds, threshold=0.5):
    """ Converts probabilities to labels

     using the [threshold] and calculates metrics.

    Parameters
    ----------
    labels
    preds
    threshold

    Returns
    -------

    """
    np.savetxt(join(cfg['paths']['dataset_root'][plat][user],
                    cfg['data']['train'] + "_" +
                    cfg['transformer']['model_type'] + '_labels.txt'),
               labels)
    np.savetxt(join(cfg['paths']['dataset_root'][plat][user],
                    cfg['data']['train'] + "_" +
                    cfg['transformer']['model_type'] + '_preds.txt'),
               preds)

    logger.info("labels:\n[{}]".format(labels))
    logger.info("preds:\n[{}]".format(preds))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    logger.info("preds with threshold [{}]:\n[{}]".format(threshold, preds))

    scores = calculate_performance_sk(labels, preds)
    logger.info(f"Scores: [{threshold}]:\n[{scores}]")

    return scores


def BERT_classifier(train_df: pd.core.frame.DataFrame,
                    test_df: pd.core.frame.DataFrame, n_classes: int = 4,
                    dataset_name: str = cfg['data']['train'],
                    model_name: str = cfg['transformer']['model_name'],
                    model_type: str = cfg['transformer']['model_type'],
                    num_epoch: int = cfg['training']['num_epoch'],
                    use_cuda: bool = cfg['model']['use_cuda']) -> (dict, dict):
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
    train_df = format_inputs(train_df)
    test_df = format_inputs(test_df)

    ## Add arguments:
    model_args = MultiLabelClassificationArgs()
    model_args.num_labels = n_classes
    model_args.no_cache = True
    model_args.no_save = True
    model_args.num_train_epochs = num_epoch
    model_args.output_dir = cfg['paths']['result_dir']
    # model_args.cache_dir = cfg['paths']['cache_dir']
    model_args.fp16 = False
    # model_args.fp16_opt_level = 'O1'
    model_args.max_seq_length = cfg['transformer']['max_seq_len']
    model_args.weight_decay = cfg['transformer']['optimizer']['weight_decay']
    model_args.learning_rate = cfg['transformer']['optimizer']['lr']
    model_args.adam_epsilon = cfg['transformer']['optimizer']['adam_epsilon']
    model_args.warmup_ratio = cfg['transformer']['optimizer']['warmup_ratio']
    model_args.warmup_steps = cfg['transformer']['optimizer']['warmup_steps']
    model_args.max_grad_norm = cfg['transformer']['optimizer']['max_grad_norm']
    model_args.train_batch_size = cfg['training']['train_batch_size']
    # model_args.gradient_accumulation_steps = cfg['transformer']['gradient_accumulation_steps']
    model_args.overwrite_output_dir = True
    model_args.eval_batch_size = cfg['training']['eval_batch_size']
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_verbose = False
    model_args.evaluate_each_epoch = True
    model_args.use_early_stopping = True
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_optimizer_and_scheduler = False
    model_args.reprocess_input_data = True
    # model_args.evaluate_during_training_steps = 3000
    model_args.save_steps = 10000
    model_args.n_gpu = 0
    model_args.threshold = 0.5

    """
    You can set config in predict(): 
        {"output_hidden_states": True} in model_args to get the hidden states.

    This will give you:
    
        all_embedding_outputs: Numpy array of shape (batch_size, sequence_length, hidden_dim)
        
        all_layer_hidden_states: Numpy array of shape (num_hidden_layers, batch_size, sequence_length, hidden_dim)
    """
    model_args.output_hidden_states = True


    args={
        'output_dir':                     cfg['paths']['result_dir'],
        'cache_dir':                      cfg['paths']['cache_dir'],
        'fp16':                           False,
        'fp16_opt_level':                 'O1',
        'max_seq_length':                 cfg['transformer']['max_seq_len'],
        'weight_decay':                   cfg['transformer']['optimizer']['weight_decay'],
        'learning_rate':                  cfg['transformer']['optimizer']['lr'],
        'adam_epsilon':                   cfg['transformer']['optimizer']['adam_epsilon'],
        'warmup_ratio':                   cfg['transformer']['optimizer']['warmup_ratio'],
        'warmup_steps':                   cfg['transformer']['optimizer']['warmup_steps'],
        'max_grad_norm':                  cfg['transformer']['optimizer']['max_grad_norm'],
        'train_batch_size':               cfg['training']['train_batch_size'],
        'gradient_accumulation_steps':    cfg['transformer']['gradient_accumulation_steps'],
        'eval_batch_size':                cfg['training']['eval_batch_size'],
        'num_train_epochs':               num_epoch,
        'evaluate_during_training':       False,
        'evaluate_during_training_steps': 3000,
        'save_steps':                     10000,
        'overwrite_output_dir':           True,
        'reprocess_input_data':           True,
        'n_gpu':                          2,
        'threshold':                      0.5
    }

    ## Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        model_type=model_type, model_name=model_name, num_labels=n_classes,
        use_cuda=use_cuda and torch.cuda.is_available(), args=args)

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

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
        # logger.info(f"id: [{example.guid}], text: [{example.text_a}]")
        miss_ids.append(example.guid)
        miss_texts.append(example.text_a)

    missed_examples = pd.DataFrame({"ids": miss_ids, "texts": miss_texts})
    logger.info(f'Misclassified examples: {missed_examples}')
    missed_examples_path = dataset_name + model_type + str(num_epoch) + "missed_examples.csv"
    logger.info(f"Saving wrongly classified examples in file: [{missed_examples_path}]")
    missed_examples.to_csv(missed_examples_path)

    # logger.info(dumps(result, indent=4))
    # with open(join(cfg['paths']['dataset_root'][plat][user],
    #                cfg['data']['train'] + "_" + model_name +
    #                '_result.json'), 'w') as f:
    #     dump(result, f)

    logger.info(f"Total training time for [{num_epoch}] with [{train_df.shape[0]}] examples: [{train_time} sec]"
                f"\nPer example: [{train_time / train_df.shape[0]} sec] for model: [{model_type}]")

    logger.info(f"Total prediction time for [{test_df.shape[0]}] examples: [{prediction_time} sec]"
                f"\nPer example: [{prediction_time / test_df.shape[0]} sec] for model: [{model_type}]")

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
                        default=cfg['data']['source']['labelled'], type=str)
    parser.add_argument("-m", "--model_name",
                        default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type",
                        default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-ne", "--num_train_epochs",
                        default=cfg['training']['num_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['model']['use_cuda'], action='store_true')

    args = parser.parse_args()

    from File_Handlers.json_handler import read_labelled_json
    from Class_mapper.FIRE16_SMERP17_map import labels_mapper

    data_dir = dataset_dir

    train_df = read_labelled_json(data_dir, args.dataset_name)
    train_df = labels_mapper(train_df)

    test_df = read_labelled_json(data_dir, cfg['data']['target']['labelled'])
    test_df = test_df.sample(frac=1)

    result, model_outputs = BERT_classifier(
        train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
        model_name=args.model_name, model_type=args.model_type,
        num_epoch=args.num_train_epochs, use_cuda=args.use_cuda)
