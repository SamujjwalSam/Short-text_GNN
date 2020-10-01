# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Constructs DGL graphs
__description__ :
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

import dgl
import nltk
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
# from nltk.tokenize import TweetTokenizer

from Utils import utils, graph_utils
from build_corpus_vocab import get_dataset_fields
from Logger.logger import logger
# from config import configuration as cfg


class DGL_Graph(object):
    """
    Creates DGL graphs

    """

    def __init__(self, dataset_df: pd.core.frame.DataFrame,
                 tokenizer: nltk.tokenize.casual.TweetTokenizer = utils.tokenizer) -> None:
        self.tokenizer = tokenizer
        # self.embeddings = embeddings
        self.dataset_df = dataset_df

        ## Process tokens:
        self.S_dataset, self.S_vocab = self.create_vocab()

        logger.info("Processed {} tokens.".format(5))

    def create_vocab(self, dataset_df=None):
        if dataset_df is None:
            dataset_df = self.dataset_df
        slab_name = "s_lab_data.csv"
        dataset_df.to_csv(slab_name)

        S_dataset, (S_fields, LABEL) = get_dataset_fields(
            csv_dir='', csv_file=slab_name, labelled_data=True, min_freq=1)

        S_vocab = {
            'freqs':       S_fields.vocab.freqs,
            'str2idx_map': dict(S_fields.vocab.stoi),
            'idx2str_map': S_fields.vocab.itos,
            'vectors':     S_fields.vocab.vectors
        }

        # logger.info("Number of tokens in corpus: [{}]".format(len(corpus)))
        logger.info("Source vocab size: [{}]".format(len(S_fields.vocab)))

        return S_dataset, S_vocab

    def create_instance_dgl_graphs(self, class_names=('0', '1', '2', '3')):
        """
        Constructs individual DGL graphs for each of the data instance

        Returns:
            graphs: An array containing DGL graphs
        """
        graphs = []
        doc_uniq_tokens = []
        labels = []

        for item in self.S_dataset.examples:
            g, doc_tokens = self.create_instance_dgl_graph(item.text)
            graphs.append(g)
            doc_uniq_tokens.append(doc_tokens)
            class_vals = []
            for cls in class_names:
                class_vals.append(int(item.__getattribute__(cls)))
            labels.append(class_vals)
        return graphs, doc_uniq_tokens, labels

    @staticmethod
    def visualize_dgl_graph(graph):
        """
        visualize single dgl graph

        Args:
            graph: dgl graph
        """
        graph_utils.visualize_dgl_graph_as_networkx(graph)

    @staticmethod
    def save_graphs(path, graphs, labels):
        labels_dict_tensor = {"glabel": torch.tensor(labels)}
        # labels_dict_tensor = torch.tensor(labels_dict)
        graph_utils.save_dgl_graphs(path, graphs, labels_dict_tensor)
        logger.info(f"Storing instance DGL graphs: {path}")

    @staticmethod
    def gen_sliding_window_instance_graph(tokens: list, token2ids_map: dict,
                                          window_size=2):
        """ Creates edge lists (sources, destinations) connecting each consecutive token_ids.

        :param tokens: List of ordered tokens
        :param token2ids_map: token text to id map
        :param window_size:
        :return:
        """
        edges_sources = []  # source edge ids
        edges_dest = []  # destination edge ids
        txt_len = len(tokens)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        j = 0
        for k in range(slide):
            txt_window = tokens[j:j + window_size]
            for i, token1 in enumerate(txt_window):
                for token2 in txt_window[i + 1:]:
                    edges_sources.append(token2ids_map[token1])
                    edges_dest.append(token2ids_map[token2])
            j = j + 1

        assert len(edges_sources) == len(edges_dest), \
            f'Source {len(edges_sources)} and Destination {len(edges_dest)}' \
            f' lengths differ.'

        return edges_sources, edges_dest

    def create_instance_dgl_graph(self, tokens: list, embeddings=None) -> (dgl.DGLGraph, list):
        """
        Create a single DGL graph

        NOTE: DGL only supports sequential node ids
        Args:
            text: Input data in string format

        Returns:
            DGL Graph: DGL graph for the input text
            :param embeddings:
            :param tokens:
        """
        # tokens = self.tokenizer(text)
        node_embedding = []  # node embedding
        node_counter = 0  # uniq ids for the tokens in the document
        token2ids_map = OrderedDict()  # Ordered token to id map for graph
        # token_ids = []  # global unique ids for the tokens

        if embeddings is None:
            embeddings = self.S_vocab['vectors']

        for token in tokens:
            try:
                token2ids_map[token]
            except KeyError:
                token2ids_map[token] = node_counter
                node_embedding.append(embeddings[self.S_vocab['str2idx_map'][token]])
                node_counter += 1
                # token_ids += [self.token2id[token]]

        ## Create consecutive tokens edge list:
        edges_sources, edges_dest = self.gen_sliding_window_instance_graph(
            tokens, token2ids_map)

        # Add nodes and edges to the graph:
        g = dgl.DGLGraph()
        g.add_nodes(len(token2ids_map))
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest))

        ## Add node embeddings to the graph:
        if embeddings is not None:
            g.ndata['emb'] = torch.stack(node_embedding)

        return g, list(token2ids_map.keys())

    def _compute_doc_embedding(self, node_id):
        """
        computes doc embedding by taking average of all word vectors in a document

        Args:
            node_id: id of the node in the graph

        Returns:
            embedding: averaged vector of all words vectors in the doc
        """
        doc_id = node_id - len(self.token2id)
        embedding = np.zeros(len(self.id2vec[0]))

        for word_id in self.docs[doc_id]:
            embedding += np.array(self.id2vec[word_id])

        embedding = embedding / len(self.docs[doc_id])
        return embedding

    def create_large_token_graph_dgl(self):
        """
        Creates a large dgl graph with tokens and documents as nodes.

        """
        g = dgl.DGLGraph()
        g.add_nodes(self.total_nodes)

        # add node data for vocab nodes
        ids = []
        embedding = []
        for id, __ in enumerate(self.token2id):
            ids += [id]
            embedding += [np.array(self.id2vec[id])]

        # add node data for doc nodes
        # at least one word is expected in the corpus:
        for id in range(len(self.token2id), self.total_nodes):
            ids += [id]
            embedding += [self._compute_doc_embedding(id)]

        g.ndata['id'] = torch.tensor(ids)
        g.ndata['emb'] = torch.tensor(embedding)

        pmi = utils.pmi(self.dataset_df)
        # add edges and edge data between vocab words in the dgl graph
        edges_sources = []
        edges_dest = []
        edge_data = []
        for tuples in pmi:
            word_pair = tuples[0]
            pmi_score = tuples[1]
            word1 = word_pair[0]
            word2 = word_pair[1]
            word1_id = self.token2id[word1]
            word2_id = self.token2id[word2]
            edges_sources += [word1_id]
            edges_dest += [word2_id]
            edge_data += [[pmi_score]]
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
                    {'weight': torch.tensor(edge_data)})

        labels = utils.get_labels(self.dataset_df)

        # add edges and edge data between documents:
        edges_sources = []
        edges_dest = []
        edge_data = []
        for i1 in range(len(labels)):
            for i2 in range(i1 + 1, len(labels)):
                doc1_id = len(self.token2id) + i1
                doc2_id = len(self.token2id) + i2
                weight = utils.iou(list(labels[i1]), list(labels[i2]))
                edges_sources += [doc1_id, doc2_id]
                edges_dest += [doc2_id, doc1_id]
                edge_data += [[weight], [weight]]
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
                    {'weight': torch.tensor(edge_data)})

        tf_idf_df = utils.tf_idf(self.dataset_df, vocab=self.token2id)
        # add edges and edge data between word and documents
        edges_sources = []
        edges_dest = []
        edge_data = []
        for index, doc_row in tf_idf_df.iterrows():
            doc_id = len(self.token2id) + index
            for word, tf_idf_value in doc_row.items():
                word_id = self.token2id[word]
                edges_sources += [doc_id]
                edges_dest += [word_id]
                edge_data += [[tf_idf_value]]
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
                    {'weight': torch.tensor(edge_data)})
        return g
