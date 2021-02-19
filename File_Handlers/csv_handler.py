# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Reads CSV file as DF using pandas.
__description__ : Details and usage.
__project__     : Tweet_GNN_inductive
__classes__     :
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

from os.path import exists, join
from pathlib import Path
import pandas as pd

from config import configuration as cfg, dataset_dir
from Logger.logger import logger


def read_csvs(data_dir=dataset_dir,
              filenames=('fire16_train', 'fire16_val', 'smerp17_test')):
    """ Reads multiple csv files.

    :param data_dir:
    :param filenames:
    :return:
    """
    dfs = []
    for filename in filenames:
        if exists(join(data_dir, filename+'.csv')):
            df = pd.read_csv(join(data_dir, filename+'.csv'), index_col=0,
                             header=0, encoding='utf-8', engine='python')
            df = df.sample(frac=1)
        else:
            df = None

        dfs.append(df)

    return pd.concat(dfs)


def read_csv(data_dir=dataset_dir, data_file='fire16_labeled_train', index_col=0,
             header=0):
    """ Reads csv file as DF.

    :param header:
    :param index_col:
    :param data_dir:
    :param data_file:
    :return:
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError("Directory [{}] not found.".format(data_dir))

    data_file = data_dir / (data_file+'.csv')
    logger.info(f"Reading csv file from [{data_file}]")
    if not data_file.exists():
        raise FileNotFoundError("File [{}] not found.".format(data_file))

    df = pd.read_csv(data_file, index_col=index_col, header=header,
                     encoding='utf-8', engine='python')
    df = df.sample(frac=1.)

    logger.info("Dataset size: [{}]".format(df.shape))
    logger.info("Few dataset samples: \n[{}]".format(df.head()))

    return df
