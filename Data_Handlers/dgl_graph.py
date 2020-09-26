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
import torch
from Utils import graph_utils
from Utils import utils
import numpy as np
from Logger.logger import logger
from config import configuration as cfg


class DGL_Graph(object):
    """
    Creates DGL graphs

    """

    def __init__(self, dataset_df, embeddings, tokenizer):
        self.total_nodes = 0
        self.id_to_vector = {}
        self.word_to_id = {}
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        words = {}
        num_nodes = 0  ## Stores the total number of docs + tokens
        self.docs = [[] for i in range(dataset_df.shape[0])]
        for index, item in dataset_df.iterrows():
            tokens = self.tokenizer(item[1])
            for token in tokens:
                try:
                    words[token]
                    self.docs[index] += [self.word_to_id[token]]
                except KeyError:
                    words[token] = 1
                    self.id_to_vector[num_nodes] = self.embeddings[token]
                    self.word_to_id[token] = num_nodes
                    self.docs[index] += [self.word_to_id[token]]
                    num_nodes += 1
        self.total_nodes = num_nodes + dataset_df.shape[0]
        self.dataframe = dataset_df
        logger.info("Processed {} tokens.".format(num_nodes))

    def create_instance_dgl_graphs(self):
        """
        Constructs individual DGL graphs for each of the data instance

        Returns:
            graphs: An array containing DGL graphs
        """
        graphs = []
        labels = []
        for _, item in self.dataframe.iterrows():
            graphs += [self.create_instance_dgl_graph(item[1])]
            labels += [item[2]]
        labels_dict = {"glabel": labels}
        return graphs, labels_dict

    def visualize_dgl_graph(self, graph):
        """
        visualize single dgl graph

        Args:
            graph: dgl graph
        """
        graph_utils.visualize_dgl_graph_as_networkx(graph)

    def save_graphs(self, path, graphs, labels_dict):
        labels_dict_tensor = {"glabel": torch.tensor(labels_dict["glabel"])}
        graph_utils.save_dgl_graphs(path, graphs, labels_dict_tensor)
        logger.info("Storing instance DGL graphs: " + cfg['paths']['data_root'])

    def create_instance_dgl_graph(self, text):
        """
        Create a single DGL graph

        NOTE: DGL only supports sequential node ids
        Args:
            text: Input data in string format

        Returns:
            DGL Graph: DGL graph for the input text
        """
        g = dgl.DGLGraph()
        tokens = self.tokenizer(text)
        node_embedding = []  # node embedding
        node_counter = 0  # uniq ids for the tokens in the document
        uniq_token_ids = {}  # ids to map token to id for the dgl graph
        token_ids = []  # global unique ids for the tokens

        for token in tokens:
            try:
                uniq_token_ids[token]
            except KeyError:
                uniq_token_ids[token] = node_counter
                node_embedding.append(self.embeddings[token])
                node_counter += 1
                token_ids += [self.word_to_id[token]]

        ## Create edge list:
        # edges_sources = []  # edge data
        # edges_dest = []  # edge data
        # for token in tokens:
        #     for child in token.children:
        #         edges_sources.append(uniq_token_ids[token])
        #         edges_dest.append(uniq_token_ids[child.text])

        # add nodes and edges to the graph:
        g.add_nodes(len(uniq_token_ids))
        g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest))

        ## Add node embeddings to the graph:
        g.ndata['emb'] = torch.tensor(node_embedding).float()

        # add token id attribute to node
        g.ndata['token_id'] = torch.tensor(token_ids).long()
        return g

    def _compute_doc_embedding(self, node_id):
        """
        computes doc embedding by taking average of all word vectors in a document

        Args:
            node_id: id of the node in the graph

        Returns:
            embedding: averaged vector of all words vectors in the doc
        """
        doc_id = node_id - len(self.word_to_id)
        embedding = np.zeros(len(self.id_to_vector[0]))

        for word_id in self.docs[doc_id]:
            embedding += np.array(self.id_to_vector[word_id])

        embedding = embedding / len(self.docs[doc_id])
        return embedding

    # def create_large_dgl_graph(self):
    #     """
    #     Creates a complete dgl graph tokens and documents as nodes
    #
    #     """
    #     g = dgl.DGLGraph()
    #     g.add_nodes(self.total_nodes)
    #
    #     # add node data for vocab nodes
    #     ids = []
    #     embedding = []
    #     for id, __ in enumerate(self.word_to_id):
    #         ids += [id]
    #         embedding += [np.array(self.id_to_vector[id])]
    #
    #     # add node data for doc nodes
    #     # at least one word is expected in the corpus:
    #     for id in range(len(self.word_to_id), self.total_nodes):
    #         ids += [id]
    #         embedding += [self._compute_doc_embedding(id)]
    #
    #     g.ndata['id'] = torch.tensor(ids)
    #     g.ndata['emb'] = torch.tensor(embedding)
    #
    #     pmi = utils.pmi(self.dataframe)
    #     # add edges and edge data betweem vocab words in the dgl graph
    #     edges_sources = []
    #     edges_dest = []
    #     edge_data = []
    #     for tuples in pmi:
    #         word_pair = tuples[0]
    #         pmi_score = tuples[1]
    #         word1 = word_pair[0]
    #         word2 = word_pair[1]
    #         word1_id = self.word_to_id[word1]
    #         word2_id = self.word_to_id[word2]
    #         edges_sources += [word1_id]
    #         edges_dest += [word2_id]
    #         edge_data += [[pmi_score]]
    #     g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
    #                 {'weight': torch.tensor(edge_data)})
    #
    #     labels = utils.get_labels(self.dataframe)
    #
    #     # add edges and edge data between documents:
    #     edges_sources = []
    #     edges_dest = []
    #     edge_data = []
    #     for i1 in range(len(labels)):
    #         for i2 in range(i1 + 1, len(labels)):
    #             doc1_id = len(self.word_to_id) + i1
    #             doc2_id = len(self.word_to_id) + i2
    #             weight = utils.iou(list(labels[i1]), list(labels[i2]))
    #             edges_sources += [doc1_id, doc2_id]
    #             edges_dest += [doc2_id, doc1_id]
    #             edge_data += [[weight], [weight]]
    #     g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
    #                 {'weight': torch.tensor(edge_data)})
    #
    #     tf_idf_df = utils.tf_idf(self.dataframe, vocab=self.word_to_id)
    #     # add edges and edge data between word and documents
    #     edges_sources = []
    #     edges_dest = []
    #     edge_data = []
    #     for index, doc_row in tf_idf_df.iterrows():
    #         doc_id = len(self.word_to_id) + index
    #         for word, tf_idf_value in doc_row.items():
    #             word_id = self.word_to_id[word]
    #             edges_sources += [doc_id]
    #             edges_dest += [word_id]
    #             edge_data += [[tf_idf_value]]
    #     g.add_edges(torch.tensor(edges_sources), torch.tensor(edges_dest),
    #                 {'weight': torch.tensor(edge_data)})
    #     return g
