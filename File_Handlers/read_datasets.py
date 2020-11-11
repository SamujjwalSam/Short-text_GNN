from os.path import join, exists
from File_Handlers.json_handler import read_labelled_json
from File_Handlers.csv_handler import read_csv
from Class_mapper.FIRE16_SMERP17_map import labels_mapper
from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Logger.logger import logger


def load_fire16(data_dir=dataset_dir, filename='fire16_labeled', data_set='train'):
    mapped_file = filename + '_4class.csv'
    if exists(join(data_dir, mapped_file)):
        # data_df = read_labelled_json(data_dir=data_dir, filename=mapped_file, data_set=data_set)
        data_df = read_tweet_csv(data_dir=data_dir,
                                 data_file=mapped_file, index_col=0,
                                 header=0)
    else:
        data_df = read_labelled_json(data_dir=data_dir, filename=filename, data_set=data_set)

        ## Match label space between two datasets:
        data_df = labels_mapper(data_df)

        # delete all rows where sum == 0
        irrelevant_rows = []
        for i, row in data_df.iterrows():
            if sum(row[1:]) < 1:
                irrelevant_rows.append(i)

        data_df = data_df.drop(irrelevant_rows)

        data_df.to_csv(join(data_dir, mapped_file))

    return data_df


def load_smerp17(data_dir=cfg["paths"]["dataset_dir"][plat][user],
                 filename='smerp17_labeled', data_set='test'):
    data_df = read_labelled_json(data_dir=data_dir, filename=filename)

    # data_df.to_csv(join(data_dir, filename+'_4class'))

    return data_df
