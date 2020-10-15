# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : PyTorch Lightning class for graph data
__description__ :  using DGL library
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
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dgl import batch as g_batch

# from dgl_dataset import GraphDataset
from .graph_dataset_dgl import GraphDataset
from config import configuration as cfg, platform as plat, username as user


class Graph_Data_Handler(pl.LightningDataModule):
    """
    PyTorch Lightning datamodule to handle training, testing and validation dataloaders
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, dataset_dir: str = cfg["paths"]["dataset_dir"][plat][user],
                 dataset_info: dict = cfg['data']) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_info = dataset_info
        self.graph_data = GraphDataset(dataset_dir=dataset_dir,
                                       dataset_file=dataset_info["source"]['labelled'])

    def batch_graphs(self, samples):
        """
        The input `samples` is a list of pairs (graph, label).
        """
        graphs, labels = map(list, zip(*samples))
        batched_graph = g_batch(graphs)
        return batched_graph, torch.tensor(labels)

    # def prepare_data(self):
    #     MNIST(os.getcwd(), train=True, download=True)
    #     MNIST(os.getcwd(), train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage='da'):
        """ Split data into train, val and test.

        stage = 'da' signifies Domain adaptation application, load target data for test.
        """
        ## Split source data to train and validation if da is True
        if stage == 'da':
            self.graph_train, self.graph_val = self.graph_data.split_data(
                test_size=cfg['data']['test_split'])

            ## Use target data for testing:
            # TODO: set target data properly:
            self.graph_test = GraphDataset(
                dataset_dir=self.dataset_dir, domain='target',
                dataset_file=self.dataset_info["target"]['labelled'])

            ## Calculate test data size:
            test_frac = 1000.0 / len(self.graph_test)

            _, self.graph_test = self.graph_test.split_data(test_size=test_frac)
        else:
            graph_train_val, self.graph_test = self.graph_data.split_data(
                test_size=cfg['data']['test_split'])

            self.graph_train, self.graph_val = graph_train_val.split_data(
                test_size=cfg['data']['val_split'])

    def train_dataloader(self, train_batch_size=cfg['training']['train_batch_size']):
        """
        Return the dataloader for each split
        """
        # Use PyTorch's DataLoader and the collate function defined above.
        return DataLoader(self.graph_train, batch_size=train_batch_size, shuffle=True,
                          collate_fn=self.batch_graphs)

    def val_dataloader(self, val_batch_size=cfg['training']['eval_batch_size']):
        return DataLoader(self.graph_val, batch_size=val_batch_size,
                          collate_fn=self.batch_graphs)

    def test_dataloader(self, test_batch_size=cfg['training']['eval_batch_size']):
        return DataLoader(self.graph_test, batch_size=test_batch_size,
                          collate_fn=self.batch_graphs)

    @property
    def num_classes(self):
        """
        Num classes is needed to initiate the model
        Therefore we return num_classes from the graph_data we defined in init
        """
        return self.graph_data.num_classes
