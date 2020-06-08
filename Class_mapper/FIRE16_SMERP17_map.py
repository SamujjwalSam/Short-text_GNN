# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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


def mapper(fire16_df, smerp17_df):
    """ Maps FIRE16 dataset labels to SMERP17 labels.

    Class maps:
        FIRE16 -> SMERP17
            0       0
            1       1
            2       0
            3       1
            4       -
            5       3
            6       2

    :param fire16_df:
    :param smerp17_df:
    """
