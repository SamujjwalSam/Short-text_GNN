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

import pickle
from datetime import datetime
from os.path import join, exists

from Logger.logger import logger


def get_date_time_tag(current_file_name=False):
    date_time = datetime.now().strftime('%Y%m%d %H%M%S')
    tag = str(date_time) + "_"
    if current_file_name:
        tag = current_file_name + "_" + str(date_time) + "_"
    return tag


date_time_tag = get_date_time_tag()


def save_pickle(data, filename, filepath='', overwrite=False, tag=False):
    """ Saves python object as pickle file.

    :param data:
    :param filename:
    :param filepath:
    :param overwrite:
    :return:
    """
    logger.debug(("Writing to pickle file: ",
                  join(filepath, filename + ".pkl")))
    if not overwrite and exists(
            join(filepath, filename + ".pkl")):
        logger.debug("File already exists and Overwrite == False.")
        return True
    try:
        if tag:
            if exists(date_time_tag + join(filepath,
                                           filename +
                                           ".pkl")):
                logger.debug(("Overwriting on pickle file: ",
                              date_time_tag + join(filepath,
                                                   filename +
                                                   ".pkl")))
            with open(date_time_tag + join(filepath,
                                           filename + ".pkl"),
                      'wb') as pkl_file:
                pickle.dump(data, pkl_file)
            pkl_file.close()
            return True
        else:
            if exists(
                    join(filepath, filename + ".pkl")):
                logger.debug(("Overwriting on pickle file: ",
                              join(filepath,
                                   filename + ".pkl")))
            with open(join(filepath, filename + ".pkl"),
                      'wb') as pkl_file:
                pickle.dump(data, pkl_file)
            pkl_file.close()
            return True
    except Exception as e:
        logger.debug(("Could not write to pickle file: ",
                      join(filepath, filename + ".pkl")))
        logger.debug(("Failure reason: ", e))
        return False


def load_pickle(filename, filepath=''):
    """ Loads pickle file from files.

    :param filename:
    :param filepath:
    :return:
    """
    try:
        if exists(join(filepath, filename + ".pkl")):
            logger.debug(("Reading pickle file: ",
                          join(filepath, filename + ".pkl")))
            with open(join(filepath, filename + ".pkl"),
                      'rb') as pkl_file:
                loaded = pickle.load(pkl_file)
            return loaded
    except Exception as e:
        logger.debug(("Could not write to pickle file:",
                      join(filepath, filename + ".pkl")))
        logger.debug(("Failure reason:", e))
        return False
