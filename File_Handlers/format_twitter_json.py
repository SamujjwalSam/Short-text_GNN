# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Convert twitter output to dataframe.
__description__ : Details and usage.
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

import argparse
import pandas as pd
from os.path import join, exists
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir
from Logger.logger import logger
from multilingual.multilingual_operations import translate_texts


def process_json(df):
    df = df[['text', 'id']].copy()
    df.text = df.text.replace({'\r':' '}, regex=True)
    # df = df.set_index('id')
    # df.to_csv(path)

    return df


def read_mixup_dataset(data_dir=dataset_dir,
                       train_name='anonymyzed_train_data_content.json',
                       val_name='anonymyzed_val_data_content.json',
                       test_name='anonymyzed_test_data_content.json'):
    val_df = pd.read_json(join(data_dir, val_name), lines=True)
    val_df = process_json(val_df)
    test_df = pd.read_json(join(data_dir, test_name), lines=True)
    test_df = process_json(test_df)
    train_df = pd.read_json(join(data_dir, train_name), lines=True)
    train_df = process_json(train_df)
    logger.info(f"Dataset Details:\n "
                f"Train size {train_df.shape}, lang counts: {train_df.lang.value_counts()}"
                f"\n Val size {val_df.shape}, lang counts: {val_df.lang.value_counts()}"
                f"\n Test size {test_df.shape}, lang counts: {test_df.lang.value_counts()}")

    return train_df, val_df, test_df


from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# lb = LabelBinarizer()
le = LabelEncoder()


def read_labels(label_filename='anonymyzed_val_data', data_dir=dataset_dir, ):
    lbl_df = pd.read_json(join(data_dir, label_filename + '.json'), orient='records')
    labels_hot = le.fit_transform(lbl_df.labels)
    lbl_df.rename(columns={'tweet_ids': 'id'}, inplace=True)
    # lbl_df = lbl_df.set_index('id')
    lbl_df['labels'] = pd.Series(labels_hot.tolist())

    return lbl_df


def read_mixup_labels():
    filenames = ['anonymyzed_train_data', 'anonymyzed_val_data', 'anonymyzed_test_data']
    lbl_dfs = []
    for filename in filenames:
        mixup_lbl_df = read_labels(filename, dataset_dir)
        mixup_lbl_df.to_csv(join(dataset_dir, filename.split('_')[1] + '_processed.csv'))
        lbl_dfs.append(mixup_lbl_df)

    return lbl_dfs


def prepare_mixup(data_dir=dataset_dir):
    filenames = ['anonymyzed_test_data', 'anonymyzed_train_data', 'anonymyzed_val_data', ]

    lbl_dfs = []
    for filename in filenames:
        lbl_df = read_labels(filename, data_dir)
        data_df = pd.read_json(join(data_dir, filename + '_content.json'), lines=True)
        data_df = process_json(data_df)
        # data_df = read_mixup_dataset(filename + '_content.json', data_dir)
        mixup_df = lbl_df.merge(data_df, on='id', how='inner')
        mixup_df = mixup_df.set_index('id')
        mixup_df = mixup_df[['text', 'labels']]
        mixup_df.to_csv(join(data_dir, 'mixup_' + filename.split('_')[1] + '.csv'))
        # lbl_dfs.append(mixup_lbl_df)


def translate_examples(df):
    non_eng_df = df[df.lang != 'en']
    non_eng_df_translated = translate_texts(non_eng_df.text.to_list())
    logger.info(non_eng_df_translated)


def main(args, ):
    prepare_mixup(data_dir=join(args.data_dir, "Multilingual-BERT-Disaster-master/Processed_Data/"),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-d", "--data_dir", type=str,
                        default=cfg["paths"]['dataset_root'][plat][user])
    parser.add_argument("-f", "--filename", type=str,
                        default="Multilingual-BERT-Disaster-master/"
                                "Processed_Data/anonymyzed_val_data_content.json",
                        help="Takes a txt file with tweet id per line.")
    parser.add_argument("-s", "--secrets_path", type=str, default="secrets.json")
    parser.add_argument("-o", "--output_dir", type=str, default=None)

    args = parser.parse_args()
    # input_filepath = join(args.data_path, args.filename)

    if args.output_dir is None:
        args.output_dir = join(args.data_dir)

    main(args)
