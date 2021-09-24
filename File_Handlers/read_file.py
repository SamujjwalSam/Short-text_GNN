import glob
from os import listdir
from os.path import join, getctime, exists, isdir

from Logger.logger import logger


def get_latest_filename(dir_path, filename_start='Glove_examcon_*'):
    list_of_files = glob.iglob(join(dir_path, filename_start))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=getctime)
    return latest_file


def is_empty_dir(dir_name='/your/path', filename_start='Glove_examcon_*'):
    list_of_files = glob.iglob(join(dir_name, filename_start))
    if not list(list_of_files):
        return True
    else:
        return False

# def is_empty_dir(dir_name='/your/path',filename_start='Glove_examcon_*'):
#     if exists(dir_name) and isdir(dir_name):
#         if not listdir(dir_name):
#             return True
#         else:
#             return False
#     else:
#         logger.warning(f"Path {dir_name} not found. Assuming empty.")
#         return True
