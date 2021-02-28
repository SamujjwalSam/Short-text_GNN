# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Contains system configuration details and other global
variables.

__description__ : Benefit: We can print the configuration on every run and
get the hyper-parameter configuration.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.3 "
__date__        : "04-03-2019"
__copyright__   : "Copyright (c) 2019, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
found in the LICENSE file in the root directory of this source tree."

__classes__     : config

__variables__   : configuration, seed, platform

__methods__     :

__last_modified__:
"""

import json
from os.path import join
from torch import cuda, device

# from Logger.logger import create_logger
# logger = create_logger(logger_name=f"{cfg['data']['name']}")

global seed
seed = 0

global configuration
configuration = {
    "data":         {
        # 'name':       'fire16_smerp17',
        # 'train':      'fire16_train',
        # 'val':        'fire16_val',
        # 'test':       'smerp17_test',
        # "source":     {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'smerp17_unlabeled'},

        # 'name':       'fire16_smerp17_source',
        # 'train':      'fire16_train',
        # 'val':        'fire16_val',
        # 'test':       'smerp17_test',
        # "source":     {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':        'smerp17_fire16',
        # 'train':       'smerp17_train',
        # 'val':         'smerp17_train',
        # 'test':        'fire16_test',
        # "source":      {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'smerp17_unlabeled'
        # },
        # "target":      {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':       'smerp17_fire16_small',
        # 'train':      'smerp17_train_75',
        # 'val':        'smerp17_val',
        # 'test':       'fire16_test',
        # "source":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'smerp17_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':       'smerp17_fire16_target',
        # 'train':      'smerp17_train',
        # 'val':        'smerp17_val',
        # 'test':       'fire16_test',
        # "source":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':       'fire16_fire16',
        # 'train':      'fire16_train',
        # 'val':        'fire16_val',
        # 'test':       'fire16_test',
        # "source":     {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'fire16_labeled',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':       'smerp17_smerp17',
        # 'train':      'smerp17_train',
        # 'val':        'smerp17_val',
        # 'test':       'smerp17_test',
        # "source":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'smerp17_unlabeled'
        # },
        # "target":     {
        #     'labelled':   'smerp17_labeled',
        #     'unlabelled': 'smerp17_unlabeled'},
        #
        # 'num_classes': 4,
        # 'class_names': ('0', '1', '2', '3'),

        'num_classes': 1,
        'class_names': ('0'),

        # 'name':        'nepal_queensland_source',
        # 'train':       '2015_Nepal_Earthquake_train',
        # 'val':         '2015_Nepal_Earthquake_dev',
        # 'test':        '2013_Queensland_Floods_test',
        # "source":      {
        #     'labelled':   '2015_Nepal_Earthquake_train',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":      {
        #     'labelled':   '2013_Queensland_Floods_test',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':        'nepal_queensland_target',
        # 'train':       '2015_Nepal_Earthquake_train',
        # 'val':         '2015_Nepal_Earthquake_dev',
        # 'test':        '2013_Queensland_Floods_test',
        # "source":      {
        #     'labelled':   '2015_Nepal_Earthquake_train',
        #     'unlabelled': 'queensland_unlabeled'
        # },
        # "target":      {
        #     'labelled':   '2013_Queensland_Floods_test',
        #     'unlabelled': 'queensland_unlabeled'},
        #
        # 'name':        'nepal_queensland',
        # 'train':       '2015_Nepal_Earthquake_train',
        # 'val':         '2015_Nepal_Earthquake_dev',
        # 'test':        '2013_Queensland_Floods_test',
        # "source":      {
        #     'labelled':   '2015_Nepal_Earthquake_train',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":      {
        #     'labelled':   '2013_Queensland_Floods_test',
        #     'unlabelled': 'queensland_unlabeled'},

        # 'name':       'queensland_nepal',
        # 'train':      '2013_Queensland_Floods_train',
        # 'val':        '2013_Queensland_Floods_dev',
        # 'test':       '2015_Nepal_Earthquake_test',
        # "source":     {
        #     'labelled':   '2013_Queensland_Floods_train',
        #     'unlabelled': 'queensland_unlabeled'
        # },
        # "target":     {
        #     'labelled':   '2015_Nepal_Earthquake_test',
        #     'unlabelled': 'fire16_unlabeled'},

        # 'name':       'queensland_nepal_source',
        # 'train':      '2013_Queensland_Floods_train',
        # 'val':        '2013_Queensland_Floods_dev',
        # 'test':       '2015_Nepal_Earthquake_test',
        # "source":     {
        #     'labelled':   '2013_Queensland_Floods_train',
        #     'unlabelled': 'queensland_unlabeled'
        # },
        # "target":     {
        #     'labelled':   '2015_Nepal_Earthquake_test',
        #     'unlabelled': 'queensland_unlabeled'},

        # 'name':       'queensland_nepal_target',
        # 'train':      '2013_Queensland_Floods_train',
        # 'val':        '2013_Queensland_Floods_dev',
        # 'test':       '2015_Nepal_Earthquake_test',
        # "source":     {
        #     'labelled':   '2013_Queensland_Floods_train',
        #     'unlabelled': 'fire16_unlabeled'
        # },
        # "target":     {
        #     'labelled':   '2015_Nepal_Earthquake_test',
        #     'unlabelled': 'fire16_unlabeled'},

        'name':        'NEQ_NEQ',
        'train':       '2015_Nepal_Earthquake_train',
        'val':         '2015_Nepal_Earthquake_dev',
        'test':        '2015_Nepal_Earthquake_test',
        "source":      {
            'labelled':   '2015_Nepal_Earthquake_train',
            'unlabelled': 'fire16_unlabeled'
        },
        "target":      {
            'labelled':   '2015_Nepal_Earthquake_train',
            'unlabelled': 'fire16_unlabeled'},

        # 'name':       'queensland_queensland',
        # 'train':      '2013_Queensland_Floods_train',
        # 'val':        '2013_Queensland_Floods_dev',
        # 'test':       '2013_Queensland_Floods_test',
        # "source":     {
        #     'labelled':   '2013_Queensland_Floods_train',
        #     'unlabelled': 'queensland_unlabeled'
        # },
        # "target":     {
        #     'labelled':   '2013_Queensland_Floods_train',
        #     'unlabelled': 'queensland_unlabeled'},

        "val_split":   0.15,
        "test_split":  0.999,
        "show_stat":   False
    },
    'pretrain':     {
        'epochs':   60,
        'save_epochs': [5, 10, 15, 25, 40, 60],
        'min_freq': 2,
        'lr':       0.005,
        'name':     'disaster_binary_pretrain',
        'files':    [
            'NEQ15',
            'IEQ12',
            'QFL13',
            'AF13_train',
            'OT13',
            'SH12'
        ],
    },

    "paths":        {
        "result_dir":    "results",
        "log_dir":       "logs",
        "cache_dir":     "cache",

        "embedding_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Embeddings",
                "cs14mtech11017": "/home/cs14mtech11017/Embeddings",
                "cs16resch01001": "/raid/cs16resch01001/Embeddings",
                ## Code path: /home/cs14resch11001/codes/MNXC
                "cs14resch11001": "/raid/ravi/pretrain"
            }
        },

        'dataset_root':  {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Datasets",
                "cs14mtech11017": "/home/cs14mtech11017/Datasets",
                "cs16resch01001": "/raid/cs16resch01001/datasets",
                "cs14resch11001": "/raid/ravi/Datasets/Extreme Classification"
            }
        }
    },

    'cuda':         {
        "use_cuda":     {
            "Windows": False,
            "OSX":     False,
            "Linux":   {
                "sam":            False,
                "cs14mtech11017": True,
                "cs16resch01001": True,
                "cs14resch11001": True
            },
        },
        "cuda_devices":     {
            "Windows": False,
            "OSX":     False,
            "Linux":   {
                "sam":            False,
                "cs14mtech11017": 1,
                "cs16resch01001": 1,
                "cs14resch11001": 7
            },
        },
    },

    "transformer":  {
        "model_type":                  "bert",
        "model_name":                  "bert-base-uncased",
        "num_folds":                   5,
        "max_seq_len":                 64,
        'gradient_accumulation_steps': 1,
        "max_vec_len":                 5000,
        "dropout":                     0.1,
        "dropout_external":            0.0,
        "clipnorm":                    2.0,
        "data_slice":                  5120,
        "normalize_inputs":            False,
        "kernel_size":                 1,
        "stride":                      1,
        "padding":                     1,
        "context":                     5,
        "classify_count":              0,
        "fce":                         True,
        "optimizer":                   {
            "optimizer_type":          "AdamW",
            "learning_rate_scheduler": "linear_warmup",
            "lr":                      3e-4,
            "lr_decay":                0.,
            "weight_decay":            0.,
            "max_grad_norm":           1.0,
            "adam_epsilon":            1e-8,
            'warmup_ratio':            0.06,
            'warmup_steps':            0.,
            "momentum":                0.9,
            "dampening":               0.9,
            "alpha":                   0.99,
            "rho":                     0.9,
            "centered":                False
        },
        "view_grads":                  False,
        "view_train_precision":        True
    },

    "model":        {
        'type':                 'LSTM',
        'mittens_iter':         500,
        "num_folds":            5,
        "max_sequence_length":  200,
        "dropout":              0.2,
        "dropout_external":     0.0,
        "clipnorm":             1.0,
        "normalize_inputs":     False,
        "kernel_size":          1,
        "stride":               1,
        "padding":              1,
        "context":              10,
        "classify_count":       0,
        "optimizer":            {
            "optimizer_type": "adam",
            "lr":             0.001,
            "lr_decay":       0,
            "weight_decay":   0,
            "momentum":       0.9,
            "dampening":      0.9,
            "alpha":          0.99,
            "rho":            0.9,
            "centered":       False
        },
        "view_grads":           False,
        "view_train_precision": True
    },

    "embeddings":   {
        'embedding_file': 'glove.6B.300d',
        'saved_emb_file': 'merged_300d',
        'emb_dim':        300,
    },

    "lstm_params":  {
        "num_layers":  2,
        "bias":        True,
        "batch_first": True,
        "bi":          True,
        "hid_size":    64,
    },

    "gnn_params":   {
        "hid_dim":     300,
        "num_heads":   2,
        "padding":     1,
        "stride":      1,
        "kernel_size": 1,
        "bias":        True,
    },

    "training":     {
        "num_epoch":        5,
        "train_batch_size": 128,
        "eval_batch_size":  256,
    },

    "prep_vecs":    {
        "max_nb_words":       20000,
        "min_word_count":     1,
        "window":             7,
        "min_freq":           1,
        "negative":           10,
        "num_chunks":         10,
        "vectorizer":         "doc2vec",
        "sample_repeat_mode": "append",
        "tfidf_avg":          False,
        "idf":                True
    },

    "text_process": {
        "encoding":         'latin-1',
        "sents_chunk_mode": "word_avg",
        "workers":          5
    },
}


class Config(object):
    """ Contains all configuration details of the project. """

    def __init__(self):
        super(Config, self).__init__()

        self.configuration = configuration

    def get_config(self):
        """

        :return:
        """
        return self.configuration

    def print_config(self, indent=4, sort=True):
        """ Prints the config. """
        print("[{}] : {}".format("Configuration",
                                 json.dumps(self.configuration,
                                            indent=indent,
                                            sort_keys=sort)))

    @staticmethod
    def get_platform():
        """ Returns dataset path based on OS.

        :return: str
        """
        import platform

        if platform.system() == 'Windows':
            return platform.system()
        elif platform.system() == 'Linux':
            return platform.system()
        else:  ## OS X returns name 'Darwin'
            return "OSX"

    @staticmethod
    def get_username():
        """
        :returns the current username.

        :return: string
        """
        try:
            import os, pwd

            username = pwd.getpwuid(os.getuid()).pw_name
        except Exception as e:
            import getpass

            username = getpass.getuser()
        # finally:
        #     username = os.environ.get('USER')

        return username


config_cls = Config()
# config_cls.print_config()

global platform
platform = config_cls.get_platform()
global username
username = config_cls.get_username()
global dataset_dir
dataset_dir = join(configuration["paths"]['dataset_root'][platform][username],
                   configuration['data']['name'])

global pretrain_dir
pretrain_dir = join(configuration['paths']['dataset_root'][platform][username],
                    configuration['pretrain']['name'])

global emb_dir
emb_dir = configuration['paths']['embedding_dir'][platform][username]

global device
device = device(f'cuda:'+str(configuration['cuda']['cuda_devices'][platform][
                                 username]) if
                cuda.is_available() else 'cpu')


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
