# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Contains system configuration details and other global
variables.

__description__ : Benefit: We can print the configuration on every run and
get the hyper-parameter configuration for each run.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "04-03-2019"
__copyright__   : "Copyright (c) 2019, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
found in the LICENSE file in the root
                  directory of this source tree."

__classes__     : config

__variables__   : configuration, seed, platform

__methods__     :

__last_modified__:
"""

import json
from Logger.logger import logger

global seed
seed = 0

global configuration
configuration = {
    "data":         {
        "dataset_name": "fire16_labeled_sample",
        "val_split":    0.1,
        "test_split":   0.3,
        "show_stat":    False
    },

    "model":        {
        "num_folds":            5,
        "max_sequence_length":  100,
        "max_vec_len":          5000,
        "dropout":              0.1,
        "dropout_external":     0.0,
        "clipnorm":             1.0,
        "data_slice":           5120,

        "g_encoder":            "cnn",
        "use_cuda":             False,
        "normalize_inputs":     False,
        "tfidf_avg":            False,
        "kernel_size":          1,
        "stride":               1,
        "padding":              1,
        "context":              10,
        "classify_count":       0,
        "fce":                  True,
        "optimizer":            {
            "optimizer_type": "adam",
            "learning_rate":  3e-4,
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

    "lstm_params":  {
        "num_layers":    2,
        "bias":          True,
        "batch_first":   True,
        "bidirectional": True,
        "hid_size":      64,
    },

    "cnn_params":   {
        "padding":     1,
        "stride":      1,
        "kernel_size": 1,
        "bias":        True,
    },

    "sampling":     {
        "num_epochs":            20,
        "num_train_epoch":       5,
        "train_batch_size":      32,
        "eval_batch_size":       64,
        "categories_per_batch":  2,
        "supports_per_category": 2,
        "targets_per_category":  2
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
        "input_size":         100,
        "tfidf_avg":          False,
        "idf":                True
    },

    "text_process": {
        "encoding":         'latin-1',
        "sents_chunk_mode": "word_avg",
        "workers":          5
    },

    "paths":        {
        "result_file":  "result.txt",
        "log_dir":      "/logs",

        "pretrain_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Embeddings",
                ## Code path: /home/cs14resch11001/codes/MNXC
                "cs14resch11001": "/raid/ravi/pretrain"
            }
        },

        "dataset_dir":  {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "OSX":     "/home/cs16resch01001/datasets/Extreme Classification",
            "Linux":   {
                "sam":            "/home/sam/Datasets/disaster_tweets",
                "cs14resch11001": "/raid/ravi/Datasets/Extreme Classification"
            }
        }
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
