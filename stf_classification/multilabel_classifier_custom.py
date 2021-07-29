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
from os import environ
from json import dumps, dump

from .multi_label_classification_model import MultiLabelClassificationModel, \
    MultiLabelClassificationArgs, ClassificationModel
from Pretrain.pretrain_BERT import get_pretrain_artifacts
from File_Handlers.csv_handler import read_csv, read_csvs
from Text_Processesor.build_corpus_vocab import get_token_embedding
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, pretrain_dir, device_id
from Metrics.metrics import calculate_performance_bin_sk
from Logger.logger import logger

if torch.cuda.is_available() and cfg['cuda']['cuda_devices'][plat][user]:
    # environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    torch.cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])
else:
    device_id = -1
# device_id, cuda_device = set_cuda_device()
device_id = cfg['cuda']['cuda_devices'][plat][user]


def format_df_cls(df: pd.core.frame.DataFrame):
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
    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    scores = calculate_performance_bin_sk(labels, preds)
    scores['dataset'] = cfg['data']['name']
    scores['epoch'] = cfg['transformer']['num_epoch']
    logger.info(f"Scores: [{threshold}]:\n[{dumps(scores, indent=4)}]")
    logger.info(f"Epoch {scores['epoch']} Test W-F1 {scores['f1_weighted'].item():1.4} Model BERT")

    return scores['f1_weighted']


def replace_bert_init_embs(model: ClassificationModel, pepoch=cfg['pretrain']['epoch']) -> None:
    """ Replace bert input tokens embeddings with pretrained embeddings.

    :param model: simpletransformer model
    :param embs_dict: Dict of token to emb (Pytorch Tensor).
    """
    logger.info('Fatching model (BERT) init embs')
    orig_embs = model.model.bert.embeddings.word_embeddings.weight
    logger.info((orig_embs, orig_embs.shape))
    orig_embs_dict = {}
    logger.info('Create token2emb map')
    for token, idx in model.tokenizer.vocab.items():
        orig_embs_dict[token] = orig_embs[idx]
    token_list = list(model.tokenizer.vocab.keys())

    logger.info('Pre-train (BERT) init embs using GCN')
    joint_vocab, token2pretrained_embs, X = get_pretrain_artifacts(orig_embs_dict, epoch=pepoch)

    logger.info('Get pretrained (BERT) embs')
    embs, _ = get_token_embedding(token_list, oov_embs=token2pretrained_embs,
                                  default_embs=orig_embs_dict, add_unk=False)
    embs = torch.nn.Parameter(embs)
    logger.info('Reassign (BERT) embs to model')
    model.model.bert.embeddings.word_embeddings.weight = embs
    logger.info((model.model.bert.embeddings.word_embeddings.weight,
                 model.model.bert.embeddings.word_embeddings.weight.shape))
    # embs = torch.nn.Embedding(embs)
    # model.model.bert.set_input_embeddings(embs)


def BERT_multilabel_classifier(
        train_df: pd.core.frame.DataFrame, val_df: pd.core.frame.DataFrame,
        test_df: pd.core.frame.DataFrame, n_classes: int = cfg['data']['num_classes'],
        dataset_name: str = cfg['data']['train'],
        model_name: str = cfg['transformer']['model_name'],
        model_type: str = cfg['transformer']['model_type'],
        num_epoch: int = cfg['transformer']['num_epoch'],
        use_cuda: bool = cfg['cuda']['use_cuda'],
        exp_name='BERT', train_all_bert=True, format_input=True,
        pretrain_embs=False, pepoch=cfg['pretrain']['epoch'],
        run_cross_tests=False) -> (dict, dict):
    """Train and Evaluation data needs to be in a Pandas Dataframe

    containing at least two columns, a 'text' and a 'labels' column. The
    `labels` column should contain multi-hot encoded lists.

    :param pepoch:
    :param pretrain_embs:
    :param format_input: Formatting for simpletransformer; not required for binary labels.
    :param train_all_bert: If whole BERT should be trained or the classifier only.
    :param exp_name:
    :param val_df:
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
    logger.info(f'{exp_name} data shapes: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}')
    if format_input:
        train_df = format_df_cls(train_df)
        val_df = format_df_cls(val_df)
        test_df = format_df_cls(test_df)

    ## Add arguments:
    model_args = MultiLabelClassificationArgs(evaluate_during_training=True)
    model_args.num_labels = n_classes
    model_args.no_cache = True
    model_args.no_save = True
    model_args.num_train_epochs = num_epoch
    model_args.output_dir = cfg['paths']['result_dir']
    # model_args.cache_dir = cfg['paths']['cache_dir']
    model_args.fp16 = False
    # model_args.fp16_opt_level = 'O1'
    model_args.max_seq_length = cfg['transformer']['max_seq_len']
    # model_args.weight_decay = cfg['transformer']['optimizer']['weight_decay']
    # model_args.learning_rate = cfg['transformer']['optimizer']['lr']
    # model_args.adam_epsilon = cfg['transformer']['optimizer']['adam_epsilon']
    # model_args.warmup_ratio = cfg['transformer']['optimizer']['warmup_ratio']
    # model_args.warmup_steps = cfg['transformer']['optimizer']['warmup_steps']
    # model_args.max_grad_norm = cfg['transformer']['optimizer']['max_grad_norm']
    model_args.train_batch_size = cfg['transformer']['train_batch_size']
    # model_args.gradient_accumulation_steps = cfg['transformer']['gradient_accumulation_steps']
    model_args.overwrite_output_dir = True
    model_args.eval_batch_size = cfg['transformer']['eval_batch_size']
    # model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_verbose = True
    # model_args.evaluate_during_training_silent = False
    model_args.evaluate_each_epoch = True
    model_args.use_early_stopping = True
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False
    model_args.save_optimizer_and_scheduler = False
    model_args.reprocess_input_data = True
    # model_args.evaluate_during_training_steps = 3000
    model_args.save_steps = 10000
    model_args.n_gpu = 1
    # model_args.learning_rate = cfg['transformer']['optimizer']['lr']
    model_args.threshold = 0.5
    model_args.early_stopping_patience = 3
    if not train_all_bert:
        logger.warning(f'Training classifier only.')
        model_args.train_custom_parameters_only = True
        model_args.custom_parameter_groups = [
            {
                "params": ["classifier.weight"],
                "lr":     1e-3,
            },
            {
                "params":       ["classifier.bias"],
                "lr":           1e-3,
                "weight_decay": 0.0,
            },
        ]
        exp_name = exp_name+"_CLSonly"

    """
    You can set config in predict(): 
        {"output_hidden_states": True} in model_args to get the hidden states.

    This will give you:
    
        all_embedding_outputs: Numpy array of shape (batch_size, sequence_length, hid_dim)
        
        all_layer_hidden_states: Numpy array of shape (num_hidden_layers, batch_size, sequence_length, hid_dim)
    """
    # model_args.output_hidden_states = True

    ## Create a MultiLabelClassificationModel
    if torch.cuda.is_available() and cfg['cuda']['cuda_devices'][plat][user]:
        model = MultiLabelClassificationModel(
            model_type=model_type, model_name=model_name, num_labels=n_classes,
            use_cuda=use_cuda and torch.cuda.is_available(), args=model_args,
            cuda_device=device_id)
    else:
        model = MultiLabelClassificationModel(
            model_type=model_type, model_name=model_name, num_labels=n_classes,
            use_cuda=use_cuda and torch.cuda.is_available(), args=model_args)

    if pretrain_embs:
        replace_bert_init_embs(model, pepoch)
        logger.warning(f'Replaced BERT embs with pepoch {pepoch}')

    # # Evaluate the model
    # result, _, _ = model.eval_model(test_df, macro_f1=macro_f1)
    # logger.info(f'BERT for experiment {exp_name+"_NOTRAIN"} Test W-F1:'
    #             f' {result["macro_f1"]:1.4} with Train {train_df.shape}, '
    #             f'Val {val_df.shape}, Test {test_df.shape}')

    ## Train the model

    # result, _, _ = model.eval_model(test_df, macro_f1=macro_f1)

    start_time = timeit.default_timer()
    model.train_model(train_df, eval_df=test_df, verbose=True, macro_f1=macro_f1)
    train_time = timeit.default_timer() - start_time
    logger.info(f'Experiment name: {exp_name}')

    ## Evaluate the model
    # result, _, _ = model.eval_model(test_df, macro_f1=macro_f1)
    # logger.info(f'BERT for experiment {exp_name+"_TRAINED"} Test W-F1:'
    #             f' {result["macro_f1"]:1.4} with Train {train_df.shape}, '
    #             f'Val {val_df.shape}, Test {test_df.shape}')
    if run_cross_tests:
        for test_data in cfg['data']['all_test_files']:
            logger.info(f'TEST data: {test_data}')
            test_df = read_csv(data_dir=pretrain_dir, data_file=test_data)
            test_df = test_df.sample(frac=1)
            # test_df["labels"] = pd.to_numeric(test_df["labels"], downcast="float")
            if format_input:
                test_df = format_df_cls(test_df)
            r, _, _ = model.eval_model(test_df, macro_f1=macro_f1)
            logger.info(f'BERT Data: {test_data} Cross W-F1: {r["macro_f1"]:1.4} size {test_df.shape}')

    # ## Evaluate the model
    # start_time = timeit.default_timer()
    # result, model_outputs, wrong_predictions = model.eval_model(
    #     test_df, macro_f1=macro_f1)
    # prediction_time = timeit.default_timer() - start_time
    # logger.info(f'BERT Test W-F1: {result["macro_f1"]:1.4}')
    # logger.info(f'Running BERT for experiment {exp_name} with Train {train_df.shape}, Val {val_df.shape},
    # Test {test_df.shape}')

    ## Analyze wrong predictions
    # logger.info("Wrong prediction count: [{}]".format(len(wrong_predictions)))
    # logger.info("Wrong predictions: ")
    # miss_ids = []
    # miss_texts = []
    # # misses = {"ids": miss_ids,"texts":miss_texts}
    # for example in wrong_predictions:
    #     # logger.info(f"id: [{example.guid}], text: [{example.text_a}]")
    #     miss_ids.append(example.guid)
    #     miss_texts.append(example.text_a)
    #
    # missed_examples = pd.DataFrame({"ids": miss_ids, "texts": miss_texts})
    # logger.info(f'Misclassified examples: {missed_examples}')
    # missed_examples_path = dataset_name + model_type + str(num_epoch) + "missed_examples.csv"
    # logger.info(f"Saving wrongly classified examples in file: [{missed_examples_path}]")
    # missed_examples.to_csv(missed_examples_path)

    # logger.info(dumps(result, indent=4))
    # with open(join(cfg['paths']['dataset_root'][plat][user],
    #                cfg['data']['train'] + "_" + model_name +
    #                '_result.json'), 'w') as f:
    #     dump(result, f)

    logger.info(f"Total training time for [{num_epoch}] with [{train_df.shape[0]}] examples: [{train_time} sec]"
                f"\nPer example: [{train_time / train_df.shape[0]} sec] for model: [{exp_name}]")

    # logger.info(f"Total prediction time for [{test_df.shape[0]}] examples: [{prediction_time} sec]"
    #             f"\nPer example: [{prediction_time / test_df.shape[0]} sec] for model: [{model_type}]")

    ## For prediction just pass a list of texts.
    # predictions, raw_outputs = model.predict(
    #     ['This thing is entirely different from the other thing. '])
    # predictions, raw_outputs = model.predict(test_df)
    # logger.debug(predictions)
    # logger.debug(raw_outputs)

    # return result, model_outputs


if __name__ == "__main__":
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

    if cfg['data']['zeroshot']:
        train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
        train_df = train_df.sample(frac=1)
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
        use_cuda=args.use_cuda, exp_name='BERT_base_zeroshot', run_cross_tests=False)

    pepochs = cfg['pretrain']['save_epochs']
    for pepoch in pepochs:  ## NEXT: Run multiple train sizes
        if cfg['data']['zeroshot']:
            train_df = read_csvs(data_dir=pretrain_dir, filenames=cfg['pretrain']['files'])
            train_df = train_df.sample(frac=1)
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
            use_cuda=args.use_cuda, exp_name='BERT_GCPD_zeroshot',
            pretrain_embs=True, pepoch=pepoch, run_cross_tests=False)
