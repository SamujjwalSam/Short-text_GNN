# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : DGL graph data handler
__description__ : DGL graph data handler
__project__     : Tweet_GNN_inductive
__classes__     : Tweet_GNN_inductive
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "31/10/20"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

from os.path import join, exists
from torch import tensor, stack
# from torch.utils.data import DataLoader
from collections import OrderedDict
from dgl import graph, add_self_loop, save_graphs, load_graphs, batch as g_batch
from dgl.data.utils import makedirs, save_info, load_info
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split

from Utils.utils import iterative_train_test_split
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Logger.logger import logger
from config import configuration as cfg, dataset_dir


class Instance_Dataset_DGL(DGLDataset):
    """ Instance graphs dataset in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    data_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self, dataset, vocab, dataset_name, graph_path=None,
                 class_names=cfg['data']['class_names'],
                 url=None, raw_dir=None, data_dir: str = dataset_dir,
                 force_reload=False, verbose=False):
        # assert dataset_name.lower() in ['cora', 'citeseer', 'pubmed']
        # if dataset_name.lower() == 'cora':
        #     dataset_name = 'cora_v2'
        self.dataset, self.vocab = dataset, vocab
        self.class_names = class_names
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        if graph_path is None:
            self.graph_path = join(self.data_dir, dataset_name + '_instance_dgl.bin')
        else:
            self.graph_path = graph_path
        self.info_path = join(self.data_dir, dataset_name + '_info.bin')
        super(Instance_Dataset_DGL, self).__init__(
            name='Instance_DGL_Dataset', url=url, raw_dir=self.data_dir, save_dir=data_dir,
            force_reload=force_reload, verbose=verbose)
        # self.graphs, self.instance_token2nodeids_map, self.labels = None, None, None
        self.num_labels = len(self.class_names)

    def process(self):
        ## Load or create graphs, labels, local and global ids:
        logger.info("Load or create graphs, labels, local and global ids.")
        if exists(self.graph_path):
            self.graphs, self.instance_graph_local_node_ids, self.labels, \
            self.instance_graph_global_node_ids = self.load_instance_dgl(self.graph_path)
        else:
            self.graphs, self.instance_graph_local_node_ids, self.labels, self.instance_graph_global_node_ids\
                = self.create_instance_dgls(dataset=self.dataset, vocab=self.vocab, class_names=self.class_names)
            self.save_instance_dgl()

    def __getitem__(self, idx):
        """ Get graph, label and global token ids by index.

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.instance_graph_local_node_ids[idx], \
               self.labels[idx], self.instance_graph_global_node_ids[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save_instance_dgl(self, graph_path=None):
        # save graphs and labels
        save_pickle(self.instance_graph_global_node_ids, self.dataset_name +
                    '_instance_graph_global_node_ids', filepath=self.data_dir)
        save_pickle(self.instance_graph_local_node_ids, self.dataset_name +
                    'instance_graph_local_node_ids', filepath=self.data_dir)
        if graph_path is None:
            graph_path = self.graph_path
        save_dgl(self.graphs, self.labels, graph_path)

    def load_instance_dgl(self, graph_path, infopath=None):
        """ Loads instance graphs.

        :param graph_path:
        :param infopath:
        :return:
        """
        if graph_path is None:
            graph_path = self.graph_path

        logger.info(f'Loading graphs from {join(self.data_dir, self.dataset_name)}')
        self.instance_graph_global_node_ids = load_pickle(self.dataset_name +
            '_instance_graph_global_node_ids', filepath=self.data_dir)
        self.instance_graph_local_node_ids = load_pickle(self.dataset_name +
            'instance_graph_local_node_ids', filepath=self.data_dir)
        self.graphs, self.labels = load_dgl(graph_path, infopath)
        return self.graphs, self.instance_graph_local_node_ids, \
               self.labels, self.instance_graph_global_node_ids

    def batch_graphs(self, samples):
        """ Batches graph data by creating a block adjacency matrix.

        :param samples:
        :return:
        """
        graphs, local_ids, labels, global_ids = map(list, zip(*samples))
        node_counts = []
        for g in graphs:
            node_counts.append(g.num_nodes())
        batched_graph = g_batch(graphs)
        return batched_graph, local_ids, stack(labels), global_ids, node_counts

    # def get_dataloader(self, train_batch_size=cfg['training']['train_batch_size']):
    #     return DataLoader((self.graphs, self.labels, self.instance_graph_global_node_ids),
    #                       batch_size=train_batch_size, shuffle=True, collate_fn=self.batch_graphs)

    def create_instance_dgls(self, dataset, vocab, class_names=cfg['data']['class_names']):
        """Create dgl for each instance in the dataset.

        :param dataset: TorchText dataset object containing tokenized examples.
        :param vocab: TorchText vocab with vectors.
        :param class_names:
        :return:
        """
        graphs = []
        instance_graph_global_node_ids = []
        instance_graph_local_node_ids = []
        labels = []

        ## Create dgl for each text instance:
        for item in dataset.examples:
            g, local_node_ids, global_node_ids = self.create_instance_dgl(item.text, vocab)
            graphs.append(g)
            instance_graph_global_node_ids.append(global_node_ids)
            instance_graph_local_node_ids.append(local_node_ids)

            class_vals = []
            for cls in class_names:
                class_vals.append(float(item.__getattribute__(cls)))

            labels.append(class_vals)

        return graphs, instance_graph_local_node_ids, tensor(labels), instance_graph_global_node_ids

    def create_instance_dgl(self, tokens: list, vocab, tokenid2vec_map=None) -> (graph, list):
        """ Given a tokenized list, returns dgl graph for that instance

        :param vocab:
        :param self:
        :param tokens:
        :param tokenid2vec_map:
        :return:
        """
        if tokenid2vec_map is None:
            tokenid2vec_map = vocab.vocab.vectors

        node_id = 0
        global_node_ids = []
        g_node_vecs = []
        instance_token2nodeids_map = OrderedDict()
        local_ids = []
        ## Get vectors for tokens:
        for token in tokens:
            global_id = vocab.vocab.stoi[token]
            global_node_ids.append(global_id)

            try:
                instance_token2nodeids_map[token]
            except KeyError:
                instance_token2nodeids_map[token] = node_id
                g_node_vecs.append(tokenid2vec_map[global_id])
                node_id += 1

            local_ids.append(instance_token2nodeids_map[token])

        ## Create edge using sliding window:
        u, v = get_sliding_edges(tokens, instance_token2nodeids_map)
        if not u or not v:
            logger.warning(f"Empty edge list u: {u}, v: {v} for {tokens}.")

        ## Create instance graph:
        g = graph((tensor(u), tensor(v)))

        ## Adding self-loops: May not be required as will be normalized:
        g = add_self_loop(g)

        ## Add node (tokens) vectors to the graph:
        if tokenid2vec_map is not None:
            g.ndata['emb'] = stack(g_node_vecs)

        return g, local_ids, global_node_ids

    def split_instance_dgls(self, graphs, labels, test_size=0.3, stratified=False, random_state=0):
        """ Splits graphs and labels to train and test.

        :param graphs: instance dgls
        :param labels: associated labels list
        :param test_size:
        :param stratified:
        :param random_state:
        :return:
        """
        if stratified is True:
            x_graphs, y_graphs, x_labels, y_labels = iterative_train_test_split(
                graphs, labels, test_size=test_size, random_state=random_state)
        else:
            x_graphs, y_graphs, x_labels, y_labels = train_test_split(
                graphs, labels, test_size=test_size, random_state=random_state)

        ## TODO: process splits:

        return x_graphs, y_graphs, x_labels, y_labels


def get_sliding_edges(tokens: list, token2ids_map: dict, window_size=2):
    """ Creates 2 list of ordered source and destination edges from tokenized text.

    Transpose of adjacency list

    :param tokens: List of ordered tokens
    :param token2ids_map: token text to id map
    :param window_size:
    :return:
    """
    # TODO: use TorchText ngrams_iterator() https://pytorch.org/text/data_utils.html
    txt_len = len(tokens)
    if window_size is None or window_size > txt_len:
        window_size = txt_len

    j = 0
    u, v = [], []
    slide = txt_len - window_size + 1
    for k in range(slide):
        txt_window = tokens[j:j + window_size]
        for i, token1 in enumerate(txt_window):
            for token2 in txt_window[i + 1:]:
                u.append(token2ids_map[token1])
                v.append(token2ids_map[token2])
        j = j + 1

    assert len(u) == len(v), f'Source {len(u)} and Destination {len(v)} lengths differ.'

    return u, v


def save_dgl(graphs, labels, graph_path, info=None, info_path=None):
    """ Saves dgl graphs, labels and other info.

    :param instance_graph_global_node_ids:
    :param info:
    :param graphs:
    :param labels:
    :param graph_path:
    :param num_classes:
    :param info_path:
    """
    # save graphs and labels
    logger.info(f'Saving graph data: {graph_path}')
    save_graphs(graph_path, graphs, {'labels': labels})
    # save other information in python dict
    if info_path is not None:
        save_info(info_path, {'info': info})


def load_dgl(graph_path, info_path=None):
    """ Loads saved dgl graphs, labels and other info.

    :param graph_path:
    :param info_path:
    :return:
    """
    # load processed data from directory graph_path
    logger.info(f'Loading graph data from: {graph_path}')
    graphs, label_dict = load_graphs(graph_path)
    labels = label_dict['labels']
    if info_path is not None:
        info = load_info(info_path)['info']
        return graphs, labels, info
    return graphs, labels


def prepare_data(dataset_file):
    df = read_datasets(dataset_name=dataset_file)
    graph_ob = DGL_Graph(df)
    graphs, doc_uniq_tokens, labels = get_instance_dgl()
    graph_file_path = dataset_file + '_graph'
    graph_ob.save_graphs(graph_file_path, graphs, labels)
