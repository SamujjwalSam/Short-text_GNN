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
from Logger.logger import logger

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
        'epochs':      100,
        'lr':          0.005,
        'name':        'disaster_binary_pretrain',
        'files':       ['2015_Nepal_Earthquake',
                        '2012_Italy_Earthquake', '2013_Queensland_Floods', 'AF13', 'OT13', 'SH12'
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
                "cs16resch01001": "/home/cs16resch01001/Embeddings",
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
                "cs16resch01001": "/home/cs16resch01001/datasets",
                "cs14resch11001": "/raid/ravi/Datasets/Extreme Classification"
            }
        }
    },

    # "transformer":  {
    #     "model_type":                  "distilbert",
    #     "model_name":                  "distilbert-base-uncased-distilled-squad",
    #     "num_folds":                   5,
    #     "max_seq_len":                 128,
    #     'gradient_accumulation_steps': 1,
    #     "max_vec_len":                 5000,
    #     "dropout":                     0.1,
    #     "dropout_external":            0.0,
    #     "clipnorm":                    1.0,
    #     "data_slice":                  5120,
    #     "use_cuda":                    {
    #         "Windows": False,
    #         "OSX":     False,
    #         "Linux":   {
    #             "sam":            False,
    #             "cs14mtech11017": True,
    #             "cs16resch01001": True,
    #             "cs14resch11001": True
    #         },
    #     },
    #     "normalize_inputs":            False,
    #     "kernel_size":                 1,
    #     "stride":                      1,
    #     "padding":                     1,
    #     "context":                     10,
    #     "classify_count":              0,
    #     "fce":                         True,
    #     "optimizer":                   {
    #         "optimizer_type":          "AdamW",
    #         "learning_rate_scheduler": "linear_warmup",
    #         "lr":                      3e-4,
    #         "lr_decay":                0.,
    #         "weight_decay":            0.,
    #         "max_grad_norm":           1.0,
    #         "adam_epsilon":            1e-8,
    #         'warmup_ratio':            0.06,
    #         'warmup_steps':            0.,
    #         "momentum":                0.9,
    #         "dampening":               0.9,
    #         "alpha":                   0.99,
    #         "rho":                     0.9,
    #         "centered":                False
    #     },
    #     "view_grads":                  False,
    #     "view_train_precision":        True
    # },

    "model":        {
        "num_folds":            5,
        "max_sequence_length":  200,
        "dropout":              0.2,
        "dropout_external":     0.0,
        "clipnorm":             1.0,
        "use_cuda":             {
            "Windows": False,
            "OSX":     False,
            "Linux":   {
                "sam":            False,
                "cs14mtech11017": True,
                "cs16resch01001": True,
                "cs14resch11001": True
            },
        },
        "normalize_inputs":     False,
        "kernel_size":          1,
        "stride":               1,
        "padding":              1,
        "context":              10,
        "classify_count":       0,
        "optimizer":            {
            "optimizer_type": "adam",
            "lr":             0.0001,
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
        "num_layers":    2,
        "bias":          True,
        "batch_first":   True,
        "bidirectional": True,
        "hid_size":      64,
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
        "num_epoch":        25,
        "num_train_epoch":  50,
        "train_batch_size": 64,
        "eval_batch_size":  256,
    },

    "prep_vecs":    {
        "max_nb_words":       20000,
        "min_word_count":     1,
        "window":             7,
        "min_count":          1,
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
        logger.info("[{}] : {}".format("Configuration",
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
config_cls.print_config()

global platform
platform = config_cls.get_platform()
global username
username = config_cls.get_username()
global dataset_dir
dataset_dir = join(configuration["paths"]['dataset_root'][platform][username],
                   configuration['data']['name'])


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
