# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Logger class for the project.
__description__ : Logs to console and file.
__project__     :
__classes__     : ColoredFormatter
__variables__   :
__methods__     :
__author__      : Samujjwal <cs16resch01001@iith.ac.in>
__version__     : ":  "
__date__        : "12/17/19"
__last_modified__:
"""

import logging
from json import dumps
from os import makedirs
from os.path import join, exists
import sys
from copy import copy
from datetime import datetime

# from logging import FileHandler
# from logging import Formatter

## Create logger
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

MAPPING = {
    'DEBUG':    37,  # white
    'INFO':     36,  # cyan
    'WARNING':  33,  # yellow
    'ERROR':    31,  # red
    'CRITICAL': 41,  # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'


class ColoredFormatter(logging.Formatter):

    def __init__(self, patern: str) -> None:
        logging.Formatter.__init__(self, patern)

    def format(self, record: logging.LogRecord) -> str:
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37)  # default white
        colored_levelname = '{0}{1}m{2}{3}'.format(PREFIX, seq, levelname,
                                                   SUFFIX)
        colored_record.levelname = colored_levelname
        return logging.Formatter.format(self, colored_record)


def create_logger(logger_name: str = 'root',
                  log_time: str = timestamp,
                  filepath: str = 'logs',
                  file_level: int = logging.DEBUG,
                  file_format: str = "%(asctime)s [%(levelname)s %("
                                     "funcName)s] (%(module)s:%(lineno)d) %("
                                     "message)s",
                  console_level: int = logging.DEBUG,
                  console_format: str = "[%(levelname)s] [%(module)s (%("
                                        "lineno)d): %(funcName)s] %(message)s",
                  color: bool = True,
                  ) -> logging.Logger:
    """

    :param logger_name:
    :param log_time:
    :param filepath:
    :param file_level:
    :param file_format:
    :param console_level:
    :param console_format:
    :param color:
    :return:
    """
    log_time = log_time + logger_name
    if not exists(filepath):
        makedirs(filepath)
    logger = logging.getLogger(logger_name)
    logger.setLevel(file_level)
    file_logger = logging.FileHandler(join(filepath, log_time + '.log'))
    file_logger.setLevel(file_level)
    file_logger.setFormatter(logging.Formatter(file_format))
    logger.addHandler(file_logger)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    if color:
        console.setFormatter(ColoredFormatter(console_format))
    else:
        console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    return logger


from config import configuration as cfg

logger_name = f"_[{cfg['data']['name']}]_[{cfg['data']['train']}]_[{cfg['data']['test']}]"
logger = create_logger(logger_name=logger_name, log_time=timestamp)
logger.info(dumps(cfg, indent=4))
logger.info("Logger created succesfully.")
