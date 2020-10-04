# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Dataset for graphs using DGL library
__description__ : Dataset class to store graphs using DGL library
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "05/08/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import dgl
import json
import numpy as np
import pandas as pd
import torch.utils.data
from pathlib import Path
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

from Utils import utils
from Utils import graph_utils
# from ..utils import TextProcessing
from File_Handlers.json_handler import read_labelled_json
from Class_mapper.FIRE16_SMERP17_map import labels_mapper
from Data_Handlers.graph_constructor_dgl import DGL_Graph
from config import configuration as cfg, platform as plat, username as user
from Logger.logger import logger


class GraphDataset(torch.utils.data.Dataset):
    """
    Class for parsing data from files and storing dataframe
    """

    def __init__(self, graphs: [dgl.DGLGraph] = None, labels: [float] = None,
                 dataset_file: [str, Path] = cfg["data"]["source"]['labelled'],
                 dataset_dir: [str, Path] = Path(cfg["paths"]["dataset_dir"][plat][user]),
                 dataset_df_name: str = "_data.csv", graph_file_name: str = "_graph.bin",
                 label_txt2id_name: str = "_label_txt2id.json", domain = 'source'
                 ) -> None:
        """
        Initializes the loader

        Args:
            dataset_dir: Path to the file containing the dataset.
            dataset_file: Filename of the dataset
            graph_file_name: path to the .bin file containing the saved DGL graph
        """
        self.domain = domain
        self.dataset_dir = dataset_dir
        dataset_df_path = Path(self.dataset_dir / (dataset_file + dataset_df_name))
        graph_file_path = Path(self.dataset_dir / (dataset_file + graph_file_name))
        label_txt2id_path = Path(self.dataset_dir / (dataset_file + label_txt2id_name))
        self.dataset_file = Path(dataset_file)

        assert (dataset_df_path.exists() or graph_file_path.exists() or
                (graphs is not None and labels is not None)),\
            "Either labels and graphs array should be given or graph_file_path \
            should be specified or dataset_path and dataset_file should be specified"

        ## if graphs and labels exist:
        if graphs is not None and labels is not None:
            self.graphs = graphs
            self.labels = labels
            self.classes = len(self.labels)

        ## Read if saved graph file exists:
        elif graph_file_path.exists() and label_txt2id_path.exists():
            # label_txt2id = self.read_label_txt2id_dict(label_txt2id_path)
            self.graphs, self.labels = graph_utils.load_dgl_graphs(graph_file_path)
            logger.info("Read graphs from " + graph_file_path)

        ## if dataframe exists
        elif label_txt2id_path.exists() and dataset_df_path.exists():
            df = pd.read_csv(dataset_df_path)
            # df, label_txt2id = self.read_datasets(dataset_df_path)
            logger.info("Read dataset from " + dataset_df_path)
            # label_txt2id = self.read_label_txt2id_dict(label_txt2id_path)
            # self.save_label_txt2id_dict(label_txt2id)
            # self.print_dataset_statistics(df, label_txt2id)

            graph_ob = DGL_Graph(df)
            self.graphs, self.labels = graph_ob.create_instance_dgl_graphs()
            graph_ob.save_graphs(graph_file_path, self.graphs, self.labels)

        ## Read and generate graphs:
        else:
            assert self.dataset_dir.exists(), "dataset_dir must exist."
            df = self.read_datasets(dataset_name=self.dataset_file)
            # df, label_txt2id = self.prune_dataset_df(df, label_txt2id)
            # label_txt2id = self.read_label_text_to_label_id_dict(label_txt2id_path)
            # self.save_dataset_df(df)
            # self.print_dataset_statistics(df, label_txt2id)
            graph_ob = DGL_Graph(df)
            self.graphs, doc_uniq_tokens, self.labels = graph_ob.create_instance_dgl_graphs()
            graph_ob.save_graphs(graph_file_path, self.graphs, self.labels)
        # atleast one document is expected
        self.classes = len(self.labels[0])

    def read_labelled_df(self, data_dir, labelled_data_name):
        """ Reads labelled data files.

        :param data_dir:
        :param labelled_data_name:
        :return:
        """
        if labelled_data_name is None:
            labelled_data_name = self.dataset_file
        ## Read labelled source data
        s_lab_df = read_labelled_json(str(data_dir), str(labelled_data_name))
        ## Match label space between two datasets:
        if self.domain == 'source':
            s_lab_df = labels_mapper(s_lab_df)

        return s_lab_df

    def read_datasets(self, dataset_name=None):
        """ Returns pandas dataframe, label to id mapping.

        INDEX           TEXT                   LABELS
          0       "Example sentence"     [-1, -2, 2, 0, 1]

         -2 -> label not present in the text
         -1 -> negative sentiment
          0 -> neutral sentiment
          1 -> positive sentiment
          2 -> ambiguous sentiment

          Label to ID mapping maps index of label list to the label name
        """
        if dataset_name is None:
            dataset_name = self.dataset_file
        # if self.dataset_info["name"] == "source":
        return self.read_labelled_df(self.dataset_dir, dataset_name)
        # elif self.dataset_info["name"] == "target":
        #     return self.read_labelled_df(self.dataset_dir, dataset_name)
        # else:
        #     logger.error("{} dataset not yet supported".format(self.dataset_info["name"]))
        #     return NotImplemented

    def save_label_txt2id_dict(self, label_txt2id):
        with open(cfg['paths']['data_root'] + self.dataset_file +
                  "_label_txt2id.json", "w") as f:
            json_dict = json.dumps(label_txt2id)
            f.write(json_dict)
        logger.info("Generated Label to ID mapping and stored at " + cfg['paths']['data_root'])

    def save_dataset_df(self, df):
        df.to_csv(cfg['paths']['data_root'] + self.dataset_file + "_dataset.csv")
        logger.info("Generated dataset and stored at " + cfg['paths']['data_root'])

    def read_label_txt2id_dict(self, label_txt2id_path):
        with open(label_txt2id_path, "r") as f:
            label_txt2id = json.load(f)
        logger.info("Read label to id mapping from " + label_txt2id_path)
        return label_txt2id

    # def print_dataset_statistics(self, df, label_txt2id):
    #
    #     utils.print_dataset_statistics(df, label_txt2id)
    #
    # def prune_dataset_df(self, df, label_txt2id):
    #
    #     df, label_txt2id = utils.prune_dataset_df(df, label_txt2id)
    #     return df, label_txt2id

    def split_data(self, test_size: float = 0.3, stratified: bool = False,
                   random_state: int = 0) -> [DGL_Graph, DGL_Graph]:
        """
        Splits dataset into train and test with optional stratified splitting
        returns 2 GraphDataset, one for train, one for test
        """
        if stratified is False:
            x_graphs, y_graphs, x_labels, y_labels = train_test_split(
                self.graphs, self.labels, test_size=test_size,
                random_state=random_state)
        else:
            sample_keys = lil_matrix(np.reshape(np.arange(len(self.graphs)),
                                                (len(self.graphs), -1)))
            labels = lil_matrix(np.array(self.labels))

            x_split, x_labels, y_split, y_labels = iterative_train_test_split(
                sample_keys, labels, test_size=test_size)

            x_labels = x_labels.todense().tolist()
            y_labels = y_labels.todense().tolist()
            x_split = x_split.todense().tolist()
            y_split = y_split.todense().tolist()

            x_graphs = list(map(lambda index: self.graphs[index[0]], x_split))
            y_graphs = list(map(lambda index: self.graphs[index[0]], y_split))

        x = GraphDataset(x_graphs, x_labels)
        y = GraphDataset(y_graphs, y_labels)

        return x, y

    @property
    def num_classes(self):
        """Number of classes."""
        return self.classes

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, one hot vector)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]
