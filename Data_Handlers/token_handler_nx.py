# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Graph construction codes
__description__ : Constructs large token graph and individual instance graphs
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

import numpy as np
import networkx as nx
# from scipy.spatial.distance import cosine
from os.path import join, exists
from torch import from_numpy, Tensor, sparse, sqrt, diag
from torch.utils.data import Dataset
from torch.nn.functional import kl_div, softmax, mse_loss, l1_loss

from Utils.utils import dot, save_graph, load_graph
from config import dataset_dir, configuration as cfg
from Logger.logger import logger


class Token_Dataset_nx(Dataset):
    """ Token graph dataset in NX. """

    def __init__(self, corpus_toks, C_vocab, dataset_name, S_vocab=None, T_vocab=None,
                 data_dir: str = dataset_dir, graph_path=None, G=None):
        super(Token_Dataset_nx, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        if G is None:
            if graph_path is None:
                self.graph_path = join(self.data_dir, dataset_name + '_token_nx.bin')
            else:
                self.graph_path = graph_path

            if exists(self.graph_path):
                self.G = self.load_graph(self.graph_path)
            else:
                if S_vocab is not None and T_vocab is not None:
                    self.G = self.create_token_graph(corpus_toks, C_vocab, S_vocab, T_vocab)
                else:
                    self.G = self.create_token_graph_pretrain(corpus_toks, C_vocab)
                self.save_graph(self.graph_path)
        else:
            self.G = G

        self.node_list = list(self.G.nodes)
        self.num_tokens = len(self.node_list)
        self.X = None

    def add_G(self, G):
        self.G = G

    @staticmethod
    def normalize_adj(A: Tensor, eps: float = 1E-9) -> Tensor:
        """ Normalize adjacency matrix for LPA:
        A = D^(-1/2) * A * D^(-1/2)
        A = softmax(A)

        :param A: adjacency matrix
        :param eps: small value
        :return:
        """
        D = sparse.sum(A, dim=0)
        D = sqrt(1.0 / (D.to_dense() + eps))
        D = diag(D).to_sparse()
        # nz_indices = torch.nonzero(D, as_tuple=False)
        # D = torch.sparse.FloatTensor(nz_indices.T, D, adj.shape)
        A = dot(D, A.to_dense()).to_sparse()
        A = dot(A, D.to_dense()).to_sparse()
        # A = self.softmax(A)

        return A

    def get_node_embeddings(self, oov_embs: dict, embs: dict, i2s: list, pretrain_embs=None):
        """ Generates embeddings in node_list order.

        :param pretrain_embs: Pretrained embeddings if exists.
        :param oov_embs: OOV embeddings generated by Mittens
        :param i2s: Combined map of id to text for S and T.
        :param embs: Default embedding (e.g. GloVe)
        :return:
        """
        emb_shape = embs[list(embs.keys())[0]].shape
        embs['<unk>'] = embs.get('<unk>', np.random.normal(size=emb_shape))
        embs['<pad>'] = embs.get('<pad>', np.zeros(emb_shape))
        embs['HTTPURL'] = embs.get('HTTPURL', np.random.normal(size=emb_shape))

        if pretrain_embs is not None:
            X = []
            for node_id in self.node_list:
                # logger.info((node_id, i2s[node_id]))
                node_txt = i2s[node_id]
                # logger.info((node_id, node_txt))
                # node_emb = oov_embs.get(node_txt, embs[node_txt])
                if node_txt in pretrain_embs:
                    node_emb = pretrain_embs[node_txt]
                elif node_txt in oov_embs:
                    node_emb = oov_embs[node_txt]
                elif node_txt in embs:
                    node_emb = embs[node_txt]
                else:
                    node_emb = embs['<unk>']
                X.append(node_emb)
        else:
            X = []
            for node_id in self.node_list:
                # logger.info((node_id, i2s[node_id]))
                node_txt = i2s[node_id]
                # logger.info((node_id, node_txt))
                # node_emb = oov_embs.get(node_txt, embs[node_txt])
                if node_txt in oov_embs:
                    node_emb = oov_embs[node_txt]
                elif node_txt in embs:
                    node_emb = embs[node_txt]
                else:
                    node_emb = embs['<unk>']

                X.append(node_emb)

        X = np.stack(X)
        X = from_numpy(X).float()

        return X

    def add_edge_weights(self, alpha: float = 0.5, clear_data=False):
        """ Calculate edge weight from occurrence values in source and target.

        NOTE: Removes occurrence and other attributes.

        :param clear_data:
        :param alpha: Decides weight between source and target occurrences.
        :return:
        """
        ## TODO: set alpha = 1 if occurrence in either source or target is 0.
        highest_cooccur = 0
        for n1, n2, edge_data in self.G.edges(data=True):
            n1_data = self.G.nodes[n1]
            n2_data = self.G.nodes[n2]
            if edge_data['s_pair'] > highest_cooccur:
                highest_cooccur = edge_data['s_pair']
                # token_pair = (n1, n2)
                logger.info(f'Maximum cooccur freq: {highest_cooccur} for '
                            f'{n1_data["node_txt"]} and {n2_data["node_txt"]}.')
            c1 = (edge_data['s_pair'] / (n1_data['s_co'] + n2_data['s_co'] + 1))
            c2 = (edge_data['t_pair'] / (n1_data['t_co'] + n2_data['t_co'] + 1))
            wt = ((1 - alpha) * c1) + (alpha * c2)
            if clear_data:
                edge_data.clear()

            self.G[n1][n2]['weight'] = wt

        if clear_data:
            for n in self.G.nodes:
                self.G.nodes[n].clear()

        return self.G

    def add_edge_weights_pretrain(self, clear_data: bool = False) -> nx.classes.graph.Graph:
        """ Calculate edge weight from occurrence values in source and target.

        NOTE: Removes occurrence and other attributes.

        :param clear_data:
        :param alpha: Decides weight between source and target occurrences.
        :return:
        """
        for n1, n2, edge_data in self.G.edges(data=True):
            if 'weight' in edge_data:
                continue
            n1_data = self.G.nodes[n1]
            n2_data = self.G.nodes[n2]
            # wt_old = (edge_data['freq'] / (n1_data['co'] + n2_data['co']))
            wt = ((edge_data['freq'] + 1) / (n1_data['co'] + n2_data['co'] + len(self.G.nodes)))
            if clear_data:
                edge_data.clear()

            self.G[n1][n2]['weight'] = wt

        if clear_data:
            for n in self.G.nodes:
                self.G.nodes[n].clear()

        self.G.weights_added = True

        return self.G

    def normalize_edge_weights(self, label_vectors: Tensor, eps=1e-9):
        """ Recalculate edge weight using LPA label vector similarity.

        NOTE: should be called after add_edge_weights().

        :param eps:
        :param label_vectors: Propagated class probability vectors per token.
        :return:
        """

        for n1, n2, edge_data in self.G.edges(data=True):
            # n1_data = self.G.nodes[n1]
            # n2_data = self.G.nodes[n2]
            # logger.debug(edge_data)
            n1_label_vec = label_vectors[n1] + eps
            n2_label_vec = label_vectors[n2] + eps
            # label_cosine = cosine_similarity(n1_label_vec, n2_label_vec).numpy()
            # label_l1 = l1_loss(n1_label_vec, n2_label_vec).numpy()
            label_mse = mse_loss(n1_label_vec, n2_label_vec).numpy()
            # label_kl = -kl_div(softmax(n1_label_vec, dim=0),
            #                    softmax(n2_label_vec, dim=0),
            #                    reduction='batchmean').item()

            self.G[n1][n2]['weight'] = edge_data['weight'] + label_mse

        return self.G

    def get_label_vectors(self, token2label_vec_map: dict, token_txt2id_map: list,
                          num_classes: int = 4, default_fill=0.5):
        """ Fetches label vectors ordered by node_list.

        :param token_txt2id_map:
        :param default_fill:
        :param num_classes: Number of classes
        :param token2label_vec_map: defaultdict of node to label vectors map
        :return:
        """
        ordered_node_embs = []
        for node in self.node_list:
            try:
                ordered_node_embs.append(token2label_vec_map[token_txt2id_map[node]])
            except KeyError:
                ordered_node_embs.append([default_fill] * num_classes)

        ordered_node_embs = np.stack(ordered_node_embs)
        ordered_node_embs = from_numpy(ordered_node_embs).float()

        return ordered_node_embs

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only token graph"
        return self.G

    def __len__(self):
        return 1

    def save_graph(self, graph_path=None):
        # save graphs and labels
        if graph_path is None:
            graph_path = self.graph_path
        save_graph(self.G, graph_path=graph_path)
        logger.info(f'Saved graph at [{graph_path}]')

    def load_graph(self, graph_path):
        if graph_path is None:
            graph_path = self.graph_path

        # load processed data from directory graph_path
        logger.info(f'Loading graph from [{graph_path}]')
        self.G = load_graph(graph_path)
        return self.G

    def get_token_adj(self):
        """ Converts NetworkX graph to torch.tensor adjacency matrix.

        :param G: nx.Graph with edge weights.
        """
        A = nx.adjacency_matrix(self.G, nodelist=self.node_list, weight='weight')
        return A

    @staticmethod
    def create_token_graph(datasets, c_vocab, s_vocab, t_vocab, window_size: int = 2):
        """ Creates a nx graph from tokenized source and target dataset.

         Use source vocab [s_vocab] for text to id mapping if exists, else use
          [t_vocab].

        :param t_vocab: Target data vocab
        :param s_vocab: Source data vocab
        :param c_vocab: Combined vocab
        :param edge_attr: Name of the edge attribute, should match with param name
         when calling add_edge().
        :param window_size: Sliding window size
        :param datasets: TorchText datasets, first source and then target.
        :return:
        """
        ## Create empty graph
        G = nx.Graph()

        ## Add token's id as node to the graph
        for token_txt, token_id in c_vocab['str2idx_map'].items():
            s_co = s_vocab['freqs'].get(token_txt, 0)
            t_co = t_vocab['freqs'].get(token_txt, 0)
            G.add_node(token_id, node_txt=token_txt, s_co=s_co, t_co=t_co)

        ## Add edges based on token co-occurrence within sliding window:
        for i, dataset in enumerate(datasets):
            for txt_toks in dataset:
                j = 0
                txt_len = len(txt_toks)
                slide = txt_len - window_size + 1
                for k in range(slide):
                    txt_window = txt_toks[j:j + window_size]
                    ## Co-occurrence in tweet:
                    occurrences = find_cooccurrences(txt_window)

                    ## Add edges with attribute:
                    for token_pair, freq in occurrences.items():
                        ## Get token ids from source if exists else from target
                        token1_id = c_vocab['str2idx_map'][token_pair[0]]
                        token2_id = c_vocab['str2idx_map'][token_pair[1]]

                        if i == 0:
                            if G.has_edge(token1_id, token2_id):
                                ##  Add value to existing edge if exists:
                                G[token1_id][token2_id]['s_pair'] += freq
                            else:  ## Add new edge if not exists and make s_pair = 0
                                G.add_edge(token1_id, token2_id, s_pair=freq,
                                           t_pair=0)
                        elif i == 1:
                            if G.has_edge(token1_id, token2_id):
                                ##  Add value to existing edge if exists:
                                G[token1_id][token2_id]['t_pair'] += freq
                            else:  ## Add new edge if not exists and make s_pair = 0
                                G.add_edge(token1_id, token2_id, s_pair=0,
                                           t_pair=freq)
                        else:
                            raise Exception(f"Unknown number [{i}] of datasets"
                                            f" provided.")

                    j = j + 1

        return G

    @staticmethod
    def create_token_graph_pretrain(dataset, c_vocab, window_size: int = 2):
        """ Creates a nx graph from tokenized dataset for pretraining.

        :param c_vocab: Combined vocab
        :param window_size: Sliding window size
        :param dataset: TorchText dataset
        :return:
        """
        ## Create empty graph
        G = nx.Graph()

        ## Add token's id as node to the graph
        for token_txt, token_id in c_vocab['str2idx_map'].items():
            co = c_vocab['freqs'].get(token_txt, 0)
            # t_co = t_vocab['freqs'].get(token_txt, 0)
            G.add_node(token_id, node_txt=token_txt, co=co)

        ## Add edges based on token co-occurrence within sliding window:
        # for i, dataset in enumerate(datasets):
        for txt_toks in dataset:
            j = 0
            txt_len = len(txt_toks)
            slide = txt_len - window_size + 1
            for k in range(slide):
                txt_window = txt_toks[j:j + window_size]
                ## Co-occurrence in tweet:
                occurrences = find_cooccurrences(txt_window)

                ## Add edges with attribute:
                for token_pair, freq in occurrences.items():
                    ## Get token ids from source if exists else from target
                    token1_id = c_vocab['str2idx_map'][token_pair[0]]
                    token2_id = c_vocab['str2idx_map'][token_pair[1]]
                    if G.has_edge(token1_id, token2_id):
                        ##  Add value to existing edge if exists:
                        G[token1_id][token2_id]['freq'] += freq
                    else:  ## Add new edge if not exists and make s_pair = 0
                        G.add_edge(token1_id, token2_id, freq=freq)

                j = j + 1

        return G


def find_cooccurrences(txt_window: list):
    edges = {}
    for i, token1 in enumerate(txt_window):
        for token2 in txt_window[i + 1:]:
            if token1 == token2: continue
            try:
                edges[(token1, token2)] += 1
            except KeyError as e:
                edges[(token1, token2)] = 1

    return edges


def create_src_tokengraph(dataset, vocab, G: nx.Graph = None,
                          window_size: int = 2):
    """ Given a corpus create a token Graph.

    Append to graph G if provided.

    :param edge_attr: Name of the edge attribute, should match with param name
     when calling add_edge().
    :param window_size: Sliding window size
    :param G:
    :param dataset: TorchText dataset
    :param vocab: TorchText field containing vocab.
    :return:
    """
    ## Create graph if not exist:
    if G is None:
        G = nx.Graph()

    ## Add token's id as node to the graph
    for token_txt, token_id in vocab['str2idx_map'].items():
        # try:
        #     token_emb = glove_embs[token_txt]
        # except KeyError:
        #     emb_shape = glove_embs[list(glove_embs.keys())[0]].shape
        #     glove_embs['<UNK>'] = np.random.uniform(low=0.5, high=0.5,
        #                                             size=emb_shape)
        #     token_emb = glove_embs['<UNK>']
        # G.add_node(token_id, node_txt=token_txt, s_co=field.vocab.freqs[
        #     token_txt], t_co=0, emb=token_emb)
        G.add_node(token_id, node_txt=token_txt, s_co=vocab['freqs'][
            token_txt], t_co=0)

    ## Add edges based on token co-occurrence within a sliding window:
    for txt_toks in dataset:
        j = 0
        txt_len = len(txt_toks)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        for k in range(slide):
            txt_window = txt_toks[j:j + window_size]
            ## Co-occurrence in tweet:
            occurrences = find_cooccurrences(txt_window)

            ## Add edges with attribute:
            for token_pair, wt in occurrences.items():
                node1 = vocab['str2idx_map'][token_pair[0]]
                node2 = vocab['str2idx_map'][token_pair[1]]
                if G.has_edge(node1, node2):
                    wt = G.get_edge_data(node1, node2)['s_pair'] + wt
                G.add_edge(node1, node2, s_pair=wt, t_pair=0)
            j = j + 1

    return G


def create_tgt_tokengraph(dataset, t_vocab, s_vocab, G: nx.Graph = None,
                          window_size: int = 2):
    """ Given a target dataset adds new nodes (occurs only in target domain)
    to existing token Graph. Update t_co count if node already exists.

     Use source vocab [s_vocab] for text to id mapping if exists, else use
      [t_vocab].

     NOTE: This should be called only after create_src_tokengraph() was called
     to create G.

    :param edge_attr: Name of the edge attribute, should match with param name
     when calling add_edge().
    :param window_size: Sliding window size
    :param G:
    :param dataset: TorchText dataset
    :param field: TorchText field containing vocab.
    :return:
    """
    ## Raise error if G not exist:
    if G is None:
        raise NotImplementedError('This method should be called only after '
                                  'create_src_tokengraph() was called to '
                                  'create G.')

    combined_s2i = s_vocab['str2idx_map']
    combined_i2s = s_vocab['idx2str_map']
    t_idx_start = len(s_vocab['str2idx_map']) + 1
    ## Add token's id (from s_vocab) as node id to the graph
    for token_str, token_id in t_vocab['str2idx_map'].items():
        if s_vocab['str2idx_map'][token_str] == 0 and token_str != '<unk>':
            # token_id = t_vocab.vocab.stoi[token_str]
            combined_s2i[token_str] = t_idx_start
            # combined_i2s[t_idx_start] = token_str
            combined_i2s.append(token_str)
            # try:
            #     token_emb = glove_embs[token_str]
            # except KeyError:
            #     token_emb = glove_embs['<unk>']
            # G.add_node(token_id, node_txt=token_str, s_co=0, t_co=token_id,
            #            emb=token_emb)
            G.add_node(t_idx_start, node_txt=token_str, s_co=0,
                       t_co=t_vocab['freqs'][token_str])
            t_idx_start = t_idx_start + 1
        # try:  ## Just add t_co value if node exists in G
        # except KeyError:  ## Create new node with s_co = 0 if node not in G
        else:
            G.node[s_vocab['str2idx_map'][token_str]]['t_co'] =\
                t_vocab['freqs'][token_str]

    for txt_toks in dataset:
        j = 0
        txt_len = len(txt_toks)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        for k in range(slide):
            txt_window = txt_toks[j:j + window_size]
            ## Co-occurrence in tweet:
            occurrences = find_cooccurrences(txt_window)

            ## Add edges with attribute:
            for token_pair, wt in occurrences.items():
                ## Get token ids from source if exists else from target
                try:
                    token1_id = s_vocab['str2idx_map'][token_pair[0]]
                except KeyError:
                    token1_id = t_vocab['str2idx_map'][token_pair[0]]
                try:
                    token2_id = s_vocab['str2idx_map'][token_pair[1]]
                except KeyError:
                    token2_id = t_vocab['str2idx_map'][token_pair[1]]

                if G.has_edge(token1_id, token2_id):
                    ##  Add value to existing edge if exists:
                    G[token1_id][token2_id]['t_pair'] += wt
                else:  ## Add new edge if not exists and make s_pair = 0
                    G.add_edge(token1_id, token2_id, s_pair=0, t_pair=wt)
            j = j + 1

    return G, combined_s2i


def generate_window_token_graph_torch2(dataset, G: nx.Graph = None,
                                       window_size: int = 2, edge_attr='s_co'):
    """ Given a corpus create a token Graph.

    Append to graph G if provided.

    :param iter: TorchText iterator
    :param window_size: Sliding window size
    :param G:
    :return:
    """
    sample_edges_txt = {}
    for txt_obj in dataset.examples:
        j = 0
        # sample_edges_txt[txt_obj.ids] = []
        txt_len = len(txt_obj.text)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        for k in range(slide):
            txt_window = txt_obj.text[j:j + window_size]
            ## Co-occurrence in tweet:
            occurrences = find_cooccurrences(txt_window)
            for nodes, wt in occurrences.items():
                try:
                    sample_edges_txt[nodes] += wt
                except KeyError as e:
                    ## Check reverse token order:
                    try:
                        sample_edges_txt[(nodes[1], nodes[0])] += wt
                    except KeyError:
                        sample_edges_txt[nodes] = wt
                    # edges[(token1, token2)] = 1
                # sample_edges_txt[nodes] = wt
            # sample_edges_txt[txt_obj.ids].append(find_cooccurrences(
            # txt_window))
            j = j + 1
            # j = j + window_size-1

    if G is None:
        G = nx.Graph()

    for nodes, edge_wt in sample_edges_txt.items():
        # for edge in txt_obj:
        #     for nodes, edge_wt in edge.items():
        G.add_edge(nodes[0], nodes[1], edge_attr=edge_wt)

    return G


def generate_window_token_graph(corpus: list, G: nx.Graph = None,
                                window_size: int = 2):
    """ Given a corpus create a token Graph.

    Append to graph G if provided.

    :param window_size: Sliding window size
    :param corpus: List of list of str.
    :param G:
    :return:
    """
    if G is None:
        G = nx.Graph()

    # for token, freq in vocab.items():
    #     G.add_node(token, s=freq[0], t=freq[1])

    sample_edges = {}
    for i, txt in enumerate(corpus):
        j = 0
        sample_edges[i] = []
        txt_len = len(txt)
        if window_size is None or window_size > txt_len:
            window_size = txt_len

        slide = txt_len - window_size + 1

        for k in range(slide):
            txt_window = txt[j:j + window_size]
            ## Co-occurrence in tweet:
            sample_edges[i].append(find_cooccurrences(txt_window))
            j = j + 1
            # j = j + window_size-1

    for weight in sample_edges.values():
        for edge in weight:
            for nodes in edge.keys():
                G.add_edge(nodes[0], nodes[1], weight=weight)

    return G


def get_k_hop_subgraph(G: nx.Graph, txt, hop: int = 0,
                       default_weight: float = 1.):
    """ Generates 0/1-hop subgraph of a tweet by collecting all the neighbor
    nodes and getting the induced subgraph.

    :param hop: Hop count
    :param G:
    :param txt: list of tokens
    :param default_weight: Edge weight for OOV node edges
    :return:
    """
    oov_nodes = []
    all_neighbors = []
    for pos, token in enumerate(txt.text):
        if hop == 0:
            all_neighbors.append(token)
        elif hop == 1:
            ## For each token, collect all the neighbors from G:
            try:
                for tok in G.neighbors(token):
                    all_neighbors.append(tok)
            except nx.exception.NetworkXError or nx.exception.NodeNotFound:
                logger.warn(f"Token [{token}] not present in large token "
                            f"graph. \nStoring for future use.")
                oov_nodes.append((pos, token))
        else:
            raise NotImplementedError("Only 0 or 1 hop is supported.")

    H = gen_complete_graph(all_neighbors, H=nx.Graph())
    # H = gen_dependency_tree(txt.text, H=nx.Graph())
    # H = gen_graph_consecutive_tokens(all_neighbors, H=nx.Graph())

    if oov_nodes:
        ## Add oov tokens to graph and connect it to other nodes.
        for pos, token in oov_nodes:
            if pos == 0:  ## if first token is oov
                H.add_edge(txt[pos], txt[pos + 1], weight=default_weight)
            elif pos == len(txt):  ## if last token if oov
                H.add_edge(txt[pos - 1], txt[pos], weight=default_weight)
            else:  ## Connect to previous and next node with oov node
                H.add_edge(txt[pos - 1], txt[pos], weight=default_weight)
                H.add_edge(txt[pos], txt[pos + 1], weight=default_weight)

    return H


def ego_graph_nbunch_window(G: nx.Graph, nbunch: list,
                            edge_attr: str = 'weight',
                            s_weight: float = 1.):
    """ Ego_graph for a bunch of nodes, adds edges among them. connects nodes
     in nbunch with original edge weight if exists, weight if not.

     Here, window_size is always 2.

    :param G:
    :param nbunch:
    :param edge_attr:
    :param s_weight:
    :return:
    """
    if len(nbunch) == 1:
        combine = nx.ego_graph(G, nbunch[0])
    else:
        combine = None
        for i in range(len(nbunch)):
            try:
                # node1 = nx.ego_graph(G, nbunch[i-1])
                node1 = nx.ego_graph(G, nbunch[i])
                try:  ## Merge 2 ego graphs
                    if combine is None:  ## New merged graph
                        combine = node1
                    else:  ## Merge with existing graph
                        combine = nx.compose(combine, node1)
                        try:
                            ## Copy edge weight if exists
                            combine[nbunch[i - 1]][nbunch[i]][edge_attr] =\
                                G[nbunch[i - 1]][nbunch[i]][edge_attr]
                        except KeyError as e:
                            ## If edge not exist, add edge with [weight].
                            combine.add_edge(nbunch[i], nbunch[i - 1],
                                             edge_attr=s_weight)
                            # combine[node1][node2][edge_attr] = weight
                except KeyError as e:
                    continue
            except nx.exception.NodeNotFound as e:
                continue
                ## TODO: Ignoring non-existing nodes for now; need to handle
                ## similar to OOV token
                # logger.info(f"Node [{nbunch[i]}] not found in G.")
                # if i > 0:
                #     G.add_edge(nbunch[i-1], nbunch[i], weight=weight)
                #     G.add_edge(nbunch[i], nbunch[i+1], weight=weight)
                # else:
                #     G.add_edge(nbunch[i], nbunch[i+1], weight=weight)

    return combine


def gen_complete_graph(txt: list, H: nx.Graph, weight: float = 1.0):
    """ Given a list of ordered tokens, generates a graph connecting each node to all other nodes.

    :param txt: list of tokens
    :param H: Empty graph object
    :param weight: edge weights to assign.
    :return:
    """
    # H = G.subgraph(txt)
    for i, token1 in enumerate(txt):
        for token2 in txt[i + 1:]:
            H.add_edge(token1, token2, weight=weight)
    return H


def generate_sample_subgraphs(txts: list, G: nx.Graph,
                              # weight: float = 1.
                              ):
    """ Given sample texts, generate subgraph keeping the sample texts
     connected.

    :param weight: Weight for edges in sample text.
    :param txts: List of texts, each containing list of tokens(nodes).
    :param G: Token graph
    """
    txts_subgraphs = {}
    for i, txt in enumerate(txts):
        # H = ego_graph_nbunch_window(G, txt)
        H = get_k_hop_subgraph(G, txt)

        # for i, txt in enumerate(txts):
        #     H = G.subgraph(txt)
        #     ## Add sample edges
        #     H = gen_complete_graph(txt, H, weight)
        #     # for i, token1 in enumerate(txt):
        #     #     for token2 in txt[i + 1:]:
        #     #         H.add_edge(token1, token2, weight=weight)

        txts_subgraphs[i] = H
    return txts_subgraphs


if __name__ == "__main__":
    txts = ['This is the first sentence.',
            'This is the second.',
            'There is no sentence in this corpus longer than this one.',
            'My dog is named Patrick.']

    from Text_Processesor.tweet_normalizer import normalizeTweet

    txts_toks = []
    for txt in txts:
        txts_toks.append(normalizeTweet(txt, return_tokens=True))

    from Text_Processesor.build_corpus_vocab import build_corpus

    corpus, vocab = build_corpus(txts_toks)

    g_ob = Token_Dataset_nx(corpus_toks, C_vocab, S_vocab, T_vocab, )
    G = g_ob.G
    logger.info(G.nodes)

    ## Testing
    test_txts = ['There sam is no sentence',
                 'My dog is named sam.']
    test_txts_toks = []
    for txt in test_txts:
        test_txts_toks.append(normalizeTweet(txt, return_tokens=True))
    # txt = ['dog', 'first', 'sam']
    # H = G.subgraph(txt).copy()
    # H = nx.node_connected_component(G, txt)
    # H = nx.node_connected_component(G, txt).copy()
    txt_h = generate_sample_subgraphs(test_txts_toks, G)
    # txt_h = ego_graph_nbunch(G, txt)

    for txt in txt_h.values():
        logger.info(txt.nodes)
        plot_graph(txt)
    logger.info("Successfully printed.")
