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
from json import dumps, dump

from simpletransformers.classification import MultiLabelClassificationModel
from simpletransformers.classification import MultiLabelClassificationArgs

from config import configuration as cfg, platform as plat, username as user
from Utils.utils import calculate_performance
from Logger.logger import logger


## TODO: Add hyperparam optimization using WandB.


def format_inputs_old(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer.

    """
    tf_df = pd.DataFrame(df[df.columns[:1]], columns=['text'])
    labels = []
    for idx, row in df[df.columns[1:]].iterrows():
        labels.append(row.to_list())
    tf_df['labels'] = pd.Series(labels, index=tf_df.index)
    return tf_df


def format_inputs(df: pd.core.frame.DataFrame):
    """ Converts the input to proper format for simpletransformer.

    """
    df['labels'] = df[df.columns[1:]].values.tolist()
    df = df[['text', 'labels']].copy()
    return df


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
                    cfg["data"]["source"]['labelled'] + "_" +
                    cfg['transformer']['model_type'] + '_labels.txt'),
               labels)
    np.savetxt(join(cfg['paths']["dataset_dir"][plat][user],
                    cfg["data"]["source"]['labelled'] + "_" +
                    cfg['transformer']['model_type'] + '_preds.txt'),
               preds)

    logger.info("labels:\n[{}]".format(labels))
    logger.info("preds:\n[{}]".format(preds))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    logger.info("preds with threshold [{}]:\n[{}]".format(threshold, preds))

    scores = calculate_performance(labels, preds)

    return scores


def BERT_classifier(train_df, test_df, n_classes=4,
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

    ## Add arguments:
    model_args = MultiLabelClassificationArgs()
    # model_args.num_labels = n_classes
    model_args.num_train_epochs = num_epoch
    model_args.output_dir = cfg['paths']['result_dir']
    model_args.cache_dir = cfg['paths']['cache_dir']
    model_args.fp16 = False
    model_args.fp16_opt_level = 'O1'
    model_args.max_seq_length = cfg['transformer']['max_seq_len']
    model_args.weight_decay = cfg['transformer']['optimizer']['weight_decay']
    model_args.learning_rate = cfg['transformer']['optimizer']['learning_rate']
    model_args.adam_epsilon = cfg['transformer']['optimizer']['adam_epsilon']
    model_args.warmup_ratio = cfg['transformer']['optimizer']['warmup_ratio']
    model_args.warmup_steps = cfg['transformer']['optimizer']['warmup_steps']
    model_args.max_grad_norm = cfg['transformer']['optimizer']['max_grad_norm']
    model_args.train_batch_size = cfg['sampling']['train_batch_size']
    model_args.gradient_accumulation_steps = cfg['transformer'][
        'gradient_accumulation_steps']
    model_args.eval_batch_size = cfg['sampling']['eval_batch_size']
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_steps = 3000
    model_args.save_steps = 10000
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.n_gpu = 2
    model_args.threshold = 0.5

    # MultiLabelClassificationArgs(
    #     adam_epsilon=1e-08,
    #     best_model_dir='outputs/best_model',
    #     cache_dir='/cache',
    #     config={},
    #     do_lower_case=False,
    #     early_stopping_consider_epochs=False,
    #     early_stopping_delta=0,
    #     early_stopping_metric='eval_loss',
    #     early_stopping_metric_minimize=True,
    #     early_stopping_patience=3,
    #     encoding=None,
    #     eval_batch_size=256,
    #     evaluate_during_training=False,
    #     evaluate_during_training_silent=True,
    #     evaluate_during_training_steps=3000,
    #     evaluate_during_training_verbose=False,
    #     fp16=False,
    #     fp16_opt_level='O1',
    #     gradient_accumulation_steps=1,
    #     learning_rate=5e-05,
    #     local_rank=-1,
    #     logging_steps=50,
    #     manual_seed=None,
    #     max_grad_norm=1.0,
    #     max_seq_length=128,
    #     multiprocessing_chunksize=500,
    #     n_gpu=2,
    #     no_cache=False,
    #     no_save=False,
    #     num_train_epochs=2,
    #     output_dir='results',
    #     overwrite_output_dir=True,
    #     process_count=54,
    #     reprocess_input_data=True,
    #     save_best_model=True,
    #     save_eval_checkpoints=True,
    #     save_model_every_epoch=True,
    #     save_steps=10000,
    #     save_optimizer_and_scheduler=True,
    #     silent=False,
    #     tensorboard_dir=None,
    #     train_batch_size=128,
    #     use_cached_eval_features=False,
    #     use_early_stopping=False,
    #     use_multiprocessing=True,
    #     wandb_kwargs={},
    #     wandb_project=None,
    #     warmup_ratio=0.06,
    #     warmup_steps=0,
    #     weight_decay=0,
    #     tie_value=1,
    #     stride=0.8,
    #     sliding_window=False,
    #     regression=False,
    #     lazy_text_column=0,
    #     lazy_text_b_column=None,
    #     lazy_text_a_column=None,
    #     lazy_labels_column=1,
    #     lazy_header_row=True,
    #     lazy_delimiter='\t',
    #     labels_list=[],
    #     labels_map={}
    #   )

    """
    You can set config in predict(): 
        {"output_hidden_states": True} in model_args to get the hidden states.

    This will give you:
    
        all_embedding_outputs: Numpy array of shape (batch_size, sequence_length, hidden_size)
        
        all_layer_hidden_states: Numpy array of shape (num_hidden_layers, batch_size, sequence_length, hidden_size)
    """
    model_args.output_hidden_states = True
    ## Create a MultiLabelClassificationModel
    model = MultiLabelClassificationModel(
        model_type=model_type, model_name=model_name,
        # num_labels=n_classes,
        use_cuda=use_cuda,
        args=model_args,
        # args={
        #     'output_dir':                     cfg['paths']['result_dir'],
        #     'cache_dir':                      cfg['paths']['cache_dir'],
        #     'fp16':                           False,
        #     'fp16_opt_level':                 'O1',
        #     'max_seq_length':                 cfg['transformer']['max_seq_len'],
        #     'weight_decay':                   cfg['transformer']['optimizer']['weight_decay'],
        #     'learning_rate':                  cfg['transformer']['optimizer']['learning_rate'],
        #     'adam_epsilon':                   cfg['transformer']['optimizer']['adam_epsilon'],
        #     'warmup_ratio':                   cfg['transformer']['optimizer']['warmup_ratio'],
        #     'warmup_steps':                   cfg['transformer']['optimizer']['warmup_steps'],
        #     'max_grad_norm':                  cfg['transformer']['optimizer']['max_grad_norm'],
        #     'train_batch_size':               cfg['sampling']['train_batch_size'],
        #     'gradient_accumulation_steps':    cfg['transformer']['gradient_accumulation_steps'],
        #     'eval_batch_size':                cfg['sampling']['eval_batch_size'],
        #     'num_train_epochs':               num_epoch,
        #     'evaluate_during_training':       False,
        #     'evaluate_during_training_steps': 3000,
        #     'save_steps':                     10000,
        #     'overwrite_output_dir':           True,
        #     'reprocess_input_data':           True,
        #     'n_gpu':                          2,
        #     'threshold':                      0.5
        # }
        )

    # if use_cuda and torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     model.to(device)

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
    missed_file = dataset_name + model_type + str(num_epoch) + \
                  "missed_samples.csv"
    logger.info("Saving wrongly classified samples in file: [{}]"
                .format(missed_file))
    missed_samples.to_csv(missed_file)

    logger.info(dumps(result, indent=4))
    with open(join(cfg['paths']["dataset_dir"][plat][user],
                   cfg["data"]["source"]['labelled'] + "_" + model_name +
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
                        default=cfg['data']['source']['labelled'], type=str)
    parser.add_argument("-m", "--model_name",
                        default=cfg['transformer']['model_name'], type=str)
    parser.add_argument("-mt", "--model_type",
                        default=cfg['transformer']['model_type'], type=str)
    parser.add_argument("-ne", "--num_train_epochs",
                        default=cfg['sampling']['num_train_epoch'], type=int)
    parser.add_argument("-c", "--use_cuda",
                        default=cfg['model']['use_cuda'], action='store_true')

    args = parser.parse_args()

    from File_Handlers.json_handler import read_labelled_json
    from Class_mapper.FIRE16_SMERP17_map import labels_mapper

    data_dir = cfg["paths"]["dataset_dir"][plat][user]

    train_df = read_labelled_json(data_dir, args.dataset_name)
    train_df = labels_mapper(train_df)

    test_df = read_labelled_json(data_dir, cfg['data']['target']['labelled'])

    result, model_outputs = BERT_classifier(
        train_df=train_df, test_df=test_df, dataset_name=args.dataset_name,
        model_name=args.model_name, model_type=args.model_type,
        num_epoch=args.num_train_epochs, use_cuda=args.use_cuda)