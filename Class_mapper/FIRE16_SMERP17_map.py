# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Maps FIRE16 7 classes to SMERP17 4 classes.
__description__ : Details and usage.
    Class maps:
        FIRE16 -> SMERP17
            0       0
            1       1
            2       0
            3       1
            4       -
            5       3
            6       2
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

import numpy as np
import pandas as pd

# from config import configuration as cfg
from Logger.logger import logger


def labels_mapper(df, class_maps: dict = None):
    """ Maps FIRE16 dataset labels to SMERP17 labels.

    -1 denotes the column to be deleted.
    Other columns are merged using logical or.

    :param class_maps:
        FIRE16 -> SMERP17
            0       0
            1       1
            2       0
            3       1
            4       -1
            5       3
            6       2
    :param df:
    """
    if class_maps is None:
        class_maps = {
            2: 0,
            3: 1,
            # 4: -1,
            5: 3,
            6: 2
        }
    logger.info(f'Mapping classes: [{class_maps}]')
    new_cols = sorted(list(class_maps.values()))
    df2 = pd.DataFrame(columns=new_cols)
    # df2 = df[df.columns.difference(new_cols)]
    # df2['text'] = df['text']
    df2.insert(loc=0, column='text', value=df['text'])
    # df2 = df[not new_cols]
    # for col in df.columns:
    for cls, mapped_cls in class_maps.items():
        df2[mapped_cls] = np.logical_or(df[cls], df[mapped_cls]) * 1
        # if mapped_cls == -1:  ## Delete column
        #     del df[cls]
        # else:  ## Merge columns using OR:
        #     df[mapped_cls] = np.logical_or(df[cls], df[mapped_cls]) * 1
        #     del df[cls]

    # df2.index = df.index
    return df2


def main():
    a = [1, 0, 1]
    b = [1, 0, 0]
    out = np.logical_or(a, b) * 1
    logger.info(out)


if __name__ == "__main__":
    main()
