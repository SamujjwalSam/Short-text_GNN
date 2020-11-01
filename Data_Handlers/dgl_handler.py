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
from torch import tensor, stack, zeros
from collections import OrderedDict
from dgl import graph, add_self_loop, save_graphs, load_graphs, batch as g_batch
from dgl.data.utils import makedirs, save_info, load_info
from dgl.data import DGLDataset
from sklearn.model_selection import train_test_split

from Utils.utils import iterative_train_test_split
from Logger.logger import logger


class Token_Dataset_DGL(DGLDataset):
    """ Token graph dataset in DGL. """

    # _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/'\
    #        'datasets/qm7b.mat'
    # _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'

    def __init__(self, dataname, graph_path=None, raw_dir=None, force_reload=False, verbose=False):
        assert dataname.lower() in ['cora', 'citeseer', 'pubmed'], f'Dataset {dataname} not supported.'
        if dataname.lower() == 'cora':
            dataname = 'cora_v2'
        # url = _get_dgl_url(self._urls[name])
        super(Token_Dataset_DGL, self).__init__(
            name=dataname + '_Token_Dataset_DGL', url=self._url, raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose)
        if graph_path is None:
            self.graph_path = join(self.save_path, dataname + '_token_dgl.bin')
        else:
            self.graph_path = graph_path
        self.info_path = join(self.save_path, dataname + '_info.bin')
        self.G, self.label = None, None

    def process(self, datasets, vocab, window_size=2):
        # process data to a list of graphs and a list of labels
        self.G, self.label = self.load_token_dgl(self.graph_path)
        self.G, self.label = self.create_token_dgl(datasets, vocab, window_size)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.G

    def __len__(self):
        return 1

    def create_token_dgl(self, datasets, c_vocab, window_size: int = 2):
        """ Creates a dgl with all unique tokens in the corpus.

            Considers tokens from both source and target.
            Use source vocab [s_vocab] for text to id mapping if exists, else use [t_vocab].

        :param t_vocab:
        :param s_vocab:
        :param c_vocab:
        :param window_size: Sliding window size
        :param G:
        :param datasets: TorchText dataset
        :return:
        """
        us, vs = [], []

        ## Add edges based on token co-occurrence within sliding window:
        for i, dataset in enumerate(datasets):
            for example in dataset:
                u, v = get_sliding_edges(example.text, c_vocab['str2idx_map'], window_size=window_size)
                us.extend(u)
                vs.extend(v)

        G = graph(data=(tensor(us), tensor(vs)))

        ## Adding self-loops:
        G = add_self_loop(G)

        ## Add node (tokens) vectors to the graph:
        if c_vocab['vectors'] is not None:
            G.ndata['emb'] = stack(c_vocab['vectors'])

        # TODO: Convert to Weighted token graph
        G.edata['w'] = self.get_edge_weights()

        return G

    def get_adj(self):
        """ Returns weighted adjacency matrix as dense tensor.

        :return:
        """
        u, v = self.G.all_edges(order='eid')
        wadj = zeros((self.G.num_nodes(), self.G.num_nodes()))
        wadj[u, v] = self.G.edata['w']
        return wadj

    def save_token_dgl(self, graph_path=None, info=False, infopath=None):
        # save graphs and labels
        if graph_path is None:
            graph_path = self.graph_path
        save_dgl(self.G, [0], graph_path, info, infopath)

    def load_token_dgl(self, graph_path, infopath=None):
        if graph_path is None:
            graph_path = self.graph_path
        # load processed data from directory graph_path
        self.G, _ = load_dgl(graph_path, infopath)
        return self.G


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
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    def __init__(self, dataname, graph_path=None, class_names=('0', '1', '2', '3'),
                 url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=False):
        assert dataname.lower() in ['cora', 'citeseer', 'pubmed']
        if dataname.lower() == 'cora':
            dataname = 'cora_v2'
        super(Instance_Dataset_DGL, self).__init__(
            name='Instance_DGL_Dataset', url=url, raw_dir=raw_dir, save_dir=save_dir,
            force_reload=force_reload, verbose=verbose)
        self.graphs, self.doc_uniq_tokens, self.labels = None, None, None
        self.class_names = class_names
        self.num_labels = len(self.class_names)
        if graph_path is None:
            self.graph_path = join(self.save_path, dataname + '_instance_dgl.bin')
        else:
            self.graph_path = graph_path
        self.info_path = join(self.save_path, dataname + '_info.bin')
        self.G, self.label = None, None

    def download(self):
        # download raw data to local disk
        pass

    def process(self, dataset, vocab):
        # === data processing skipped ===
        mat_path = self.raw_path + '.bin'
        # process data to a list of graphs and a list of labels
        self.graphs, self.labels = self.load(mat_path)

        self.graphs, self.doc_uniq_tokens, self.labels = self.create_instance_dgls(
            dataset, vocab, class_names=self.class_names)

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save_instance_dgl(self, graph_path=None, info=False, infopath=None):
        # save graphs and labels
        if graph_path is None:
            graph_path = self.graph_path
        save_dgl(self.graphs, self.labels, graph_path, info, infopath)

    def load_instance_dgl(self, graph_path, infopath=None):
        if graph_path is None:
            graph_path = self.graph_path
        # load processed data from directory graph_path
        self.graphs, self.labels = load_dgl(graph_path, infopath)
        return self.graphs, self.labels

    def batch_graphs(self, samples):
        """ Batches graph data by creating a block adjacency matrix.

        :param samples:
        :return:
        """
        graphs, labels = map(list, zip(*samples))
        batched_graph = g_batch(graphs)
        return batched_graph, tensor(labels)

    def create_instance_dgl(self, vocab, tokens: list, token2vec_map=None) -> (graph, list):
        """ Given a tokenized list, returns dgl graph for that instance.

        :param vocab:
        :param self:
        :param tokens:
        :param token2vec_map:
        :return:
        """
        if token2vec_map is None:
            token2vec_map = vocab['vectors']

        node_ids = 0
        graph_tokens_vec = []
        token2ids_map = OrderedDict()

        ## Get vectors for tokens:
        for token in tokens:
            try:
                token2ids_map[token]
            except KeyError:
                token2ids_map[token] = node_ids
                graph_tokens_vec.append(token2vec_map[vocab['str2idx_map'][token]])
                node_ids += 1
                # token_ids += [token2id[token]]

        ## Create edge using sliding window:
        u, v = get_sliding_edges(tokens, token2ids_map)

        ## Create instance graph:
        g = graph((tensor(u), tensor(v)))

        ## Adding self-loops:
        g = add_self_loop(g)

        ## Add node (tokens) vectors to the graph:
        if token2vec_map is not None:
            g.ndata['emb'] = stack(graph_tokens_vec)

        return g, list(token2ids_map.keys())

    def create_instance_dgls(self, labelled_instances, vocab, class_names=('0', '1', '2', '3')):
        """ Create dgl for each instance in the dataset.

        :param labelled_instances: TorchText dataset object containing tokenized examples.
        :param vocab: TorchText vocab with vectors.
        :param class_names:
        :return:
        """
        graphs = []
        doc_uniq_tokens = []
        labels = []

        ## Create dgl for each text instance:
        for item in labelled_instances.examples:
            g, doc_tokens = self.create_instance_dgl(vocab, item.text)
            graphs.append(g)
            doc_uniq_tokens.append(doc_tokens)

            class_vals = []
            for cls in class_names:
                class_vals.append(float(item.__getattribute__(cls)))

            labels.append(class_vals)

        return graphs, doc_uniq_tokens, labels

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

    :param info:
    :param graphs:
    :param labels:
    :param graph_path:
    :param num_classes:
    :param info_path:
    """
    # save graphs and labels
    if not exists(graph_path):
        makedirs(graph_path)
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
