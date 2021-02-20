# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Contrastive pretraining script
__description__ : Details and usage.
__project__     : WSCP
__classes__     : WSCP
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "20/01/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import random
import numpy as np
import pandas as pd
from torch import cuda, save, load, device, manual_seed, backends, from_numpy
from torch.utils.data import Dataset
from networkx import adjacency_matrix
from os import environ
from os.path import join, exists
from sklearn.preprocessing import MultiLabelBinarizer

from Text_Encoder.TextEncoder import train_w2v
from Pretrainer.mlp_trainer import mlp_trainer
from Pretrainer.gcn_trainer import gcn_trainer
from Utils.utils import load_graph, sp_coo2torch_coo, get_token2pretrained_embs
from File_Handlers.csv_handler import read_csv
from File_Handlers.json_handler import save_json, read_json
from File_Handlers.pkl_handler import save_pickle, load_pickle
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Text_Encoder.finetune_static_embeddings import glove2dict, train_mittens,\
    calculate_cooccurrence_mat, preprocess_and_find_oov2, create_clean_corpus
from config import configuration as cfg, pretrain_dir
from Logger.logger import logger

## Enable multi GPU cuda environment:
device = device('cuda' if cuda.is_available() else 'cpu')
if cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda.set_device(0)


def set_all_seeds(seed=0):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    backends.cudnn.deterministic = True


set_all_seeds(1)

data_filenames = cfg['pretrain']['files']
# data_filenames = cfg['data']['files']
# pretrain_dir = join(cfg['paths']['dataset_root'][plat][user], cfg['pretrain']['name'])
# data_filename = cfg['data']['name']
data_filename = cfg['pretrain']['name']
joint_path = join(pretrain_dir, data_filename + "_multihot.csv")
pos_path = join(pretrain_dir, data_filename + "_pos.csv")
neg_path = join(pretrain_dir, data_filename + "_neg.csv")

glove_embs = glove2dict()


def read_dataset(dataname, data_dir=pretrain_dir):
    ## TODO: Read all binary disaster datasets:
    df = read_csv(data_dir, dataname)
    df_pos = df[df['labels'] == 1]
    df_neg = df[df['labels'] == 0]
    # df = df.sort_values(by='labels')
    # cls_counts = df.labels.value_counts()
    # logger.info(cls_counts)
    # df_pos = df.iloc[cls_counts[0]:]
    # df_neg = df.iloc[:cls_counts[0]]

    pos_path = join(data_dir, dataname + "_pos.csv")
    # df_pos.to_csv(pos_path)
    neg_path = join(data_dir, dataname + "_neg.csv")
    # df_neg.to_csv(neg_path)

    labels = [[x] for x in df.labels.to_list()]
    labels_hot = mlb.fit_transform(labels)

    df['0'] = labels_hot[:, 0]
    df['1'] = labels_hot[:, 1]
    df = df.drop(columns=['labels'], axis=0)
    df.to_csv(joint_path)

    return df_pos, df_neg, df, joint_path, pos_path, neg_path


mlb = MultiLabelBinarizer()


def read_input_files(joint_path=join(pretrain_dir, data_filename + "_multihot.csv"),
                     pos_path=join(pretrain_dir, data_filename + "_pos.csv"),
                     neg_path=join(pretrain_dir, data_filename + "_neg.csv")):
    ## Read multiple data files:
    df_all = pd.DataFrame()
    df_all_pos = pd.DataFrame()
    df_all_neg = pd.DataFrame()
    for data_file in data_filenames:
        df_pos, df_neg, df, _, _, _ = read_dataset(data_file)
        df_all = pd.concat([df_all, df])
        df_all_pos = pd.concat([df_all_pos, df_pos])
        df_all_neg = pd.concat([df_all_neg, df_neg])

    df_all.to_csv(joint_path)
    df_all_pos.to_csv(pos_path)
    df_all_neg.to_csv(neg_path)


def calculate_vocab_overlap(task_tokens, pretrain_tokens):
    """ Calculate vocab overlap between pretrain and task:"""
    # joint_set = set(joint_vocab['str2idx_map'].keys())
    # task_tokens = set(finetune_vocab['str2idx_map'].keys())
    overlap = pretrain_tokens.intersection(task_tokens)
    finetune_vocab_percent = (len(overlap) / len(task_tokens)) * 100
    logger.info(f'Token overlap between pretrain and fine-tune vocab:{len(overlap)}')
    logger.info(f'Token overlap percent as fine-tune vocab:{finetune_vocab_percent}')


def get_pretrain_dataset(C, G, G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab):
    """ Fetch N+(c) and N-(c) for each token in C: """
    dataset = []
    ignored_tokens = set()
    for i, token in enumerate(C):
        N_pos = []
        N_pos_wt_dict = {}
        N_pos_txt = []
        for n in G_pos.G.neighbors(pos_vocab['str2idx_map'][token]):
            n_txt = pos_vocab['idx2str_map'][n]
            n_id = joint_vocab['str2idx_map'][n_txt]

            ## ## Use pos_G weight:
            wt = G_pos.G[pos_vocab['str2idx_map'][token]][pos_vocab['str2idx_map'][n_txt]]
            N_pos.append(n_id)
            N_pos_wt_dict[n_id] = wt
            N_pos_txt.append(n_txt)

        N_neg = []
        N_neg_wt_dict = {}
        N_neg_txt = []
        for n in G_neg.G.neighbors(neg_vocab['str2idx_map'][token]):
            n_txt = neg_vocab['idx2str_map'][n]
            n_id = joint_vocab['str2idx_map'][n_txt]

            ## Use neg_G weight:
            wt = G_neg.G[neg_vocab['str2idx_map'][token]][neg_vocab['str2idx_map'][n_txt]]
            N_neg.append(n_id)
            N_neg_wt_dict[n_id] = wt
            N_neg_txt.append(n_txt)

        N_pos = set(N_pos)
        N_neg = set(N_neg)
        overlap = N_pos.intersection(N_neg)
        N_pos_txt = set(N_pos_txt)
        N_neg_txt = set(N_neg_txt)
        overlap_txt = N_pos_txt.intersection(N_neg_txt)
        if len(overlap) > 0:
            ## Remove common neighbors:
            ## TODO: What happens if tokens are left as pos neighbors?
            # N_pos = N_pos - overlap
            N_neg = N_neg - overlap
            # if len(overlap_txt) > 2:
            #     logger.debug((token, N_pos_txt, N_neg_txt, overlap_txt))

        N_pos = list(N_pos)
        N_neg = list(N_neg)
        N_pos_wt = []
        for node_id in N_pos:
            N_pos_wt.append(N_pos_wt_dict[node_id]['weight'])
        N_neg_wt = []
        for node_id in N_neg:
            N_neg_wt.append(N_neg_wt_dict[node_id]['weight'])

        if len(N_neg) == 0:
            # logger.warning(f'Token {token} has 0 NEG (-) neighbors.')
            ignored_tokens.add(token)
            continue

        if len(N_pos) == 0:
            # logger.warning(f'Token {token} has 0 POS (+) neighbors.')
            ignored_tokens.add(token)
            continue

        dataset.append((joint_vocab['str2idx_map'][token], N_pos, N_pos_wt,
                        N_neg, N_neg_wt))

    return dataset, ignored_tokens


def get_vocab_data(path=join(pretrain_dir, data_filename + "_multihot.csv"),
                   name='_joint', glove_embs=None, min_freq=cfg['pretrain']['min_freq']):
    if exists(join(pretrain_dir, data_filename + name + '_corpus_toks.json'))\
            and exists(join(pretrain_dir, data_filename + name + '_vocab.json')):
        vocab = read_json(join(pretrain_dir, data_filename + name + '_vocab'))
        oov_high_freqs = read_json(join(pretrain_dir, data_filename + name + '_oov_high_freqs'))
        corpus_toks = read_json(join(pretrain_dir, data_filename + name + '_corpus_toks'),
                                convert_ordereddict=False)
        corpus_strs = read_json(join(pretrain_dir, data_filename + name + '_corpus_strs'),
                                convert_ordereddict=False)
    else:
        ## TODO: Read pretraining data files:
        read_input_files()
        dataset, (fields, LABEL) = get_dataset_fields(
            csv_dir='', csv_file=path, min_freq=min_freq)

        vocab = {
            'freqs':       fields.vocab.freqs,
            'str2idx_map': dict(fields.vocab.stoi),
            'idx2str_map': fields.vocab.itos,
        }

        if glove_embs is None:
            glove_embs = glove2dict()

        oov_high_freqs, glove_low_freqs, oov_low_freqs =\
            preprocess_and_find_oov2(vocab, glove_embs=glove_embs, labelled_vocab_set=set(
                vocab['str2idx_map'].keys()), add_glove_tokens_back=False)

        corpus_strs, corpus_toks, ignored_examples =\
            create_clean_corpus(dataset, low_oov_ignored=oov_low_freqs)

        save_json(oov_high_freqs, join(pretrain_dir, data_filename + name + '_oov_high_freqs'))
        save_json(vocab, join(pretrain_dir, data_filename + name + '_vocab'))
        save_json(corpus_strs, join(pretrain_dir, data_filename + name + '_corpus_strs'))
        save_json(corpus_toks, join(pretrain_dir, data_filename + name + '_corpus_toks'))

    return vocab, corpus_toks, corpus_strs, oov_high_freqs


class Pretrain_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int):
        """ Get token id along with its pos and neg neighbor indices.

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (int, list[int], list[int])
        """
        return self.dataset[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.dataset)


def get_graph_inputs(G, oov_embs, joint_vocab, G_node_list=None, glove_embs=None, prepare_G=True):
    logger.info('Accessing graph node embeddings:')
    emb_filename = join(pretrain_dir, data_filename + "_emb.pt")
    if exists(emb_filename):
        X = load(emb_filename)
    else:
        logger.info('Get node embeddings from graph')
        if glove_embs is None:
            glove_embs = glove2dict()
        X = G.get_node_embeddings(oov_embs, glove_embs, joint_vocab['idx2str_map'])
        # X = sp_coo2torch_coo(X)
        save(X, emb_filename)

    logger.info('Accessing adjacency matrix')

    if G_node_list is None:
        G_node_list = list(G.G.nodes)

    adj = None
    ## Create adjacency matrix if required:
    if prepare_G:
        adj = adjacency_matrix(G.G, nodelist=G_node_list, weight='weight')
        adj = sp_coo2torch_coo(adj)
        logger.info('Normalize token graph:')
        adj = G.normalize_adj(adj)

    return adj, X


def get_graph_and_dataset(limit_dataset=None):
    # df, joint_path, pos_path, neg_path = read_dataset(pretrain_dir, data_filename)
    glove_embs = glove2dict()
    # oov_emb_filename = data_filename + '_OOV_vectors_dict'
    oov_emb_filename = 'disaster_binary_pretrain_OOV_vectors_dict'
    joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
        get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs)
    if exists(join(pretrain_dir, oov_emb_filename + '.pkl')):
        oov_embs = load_pickle(filepath=pretrain_dir, filename=oov_emb_filename)
    else:
        ## Create new embeddings for OOV tokens:
        if exists(join(pretrain_dir, oov_emb_filename + '.pkl')):
            logger.info('Reading OOV embeddings:')
            oov_embs = load_pickle(filepath=pretrain_dir, filename=oov_emb_filename)
        else:
            logger.info('Create OOV embeddings using Mittens:')
            high_oov_tokens_list = list(joint_oov_high_freqs.keys())
            # c_corpus = corpus[0] + corpus[1]
            oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens_list, joint_corpus_strs)
            oov_embs = train_mittens(oov_mat_coo, high_oov_tokens_list, glove_embs, max_iter=200)
            save_pickle(oov_embs, filepath=pretrain_dir, filename=oov_emb_filename, overwrite=True)

    G = Token_Dataset_nx(joint_corpus_toks, joint_vocab, dataset_name=joint_path[:-12])

    G.add_edge_weights_pretrain()
    # num_tokens = G.num_tokens
    G_node_list = list(G.G.nodes)
    logger.info(f"Number of nodes {len(G_node_list)} and edges {len(G.G.edges)} in POS token graph")
    # token2label_vec_map = freq_tokens_per_class(df, normalize=False)

    ## Create G+:
    pos_vocab, pos_corpus_toks, pos_corpus_strs, pos_oov_high_freqs =\
        get_vocab_data(pos_path, name='_pos', glove_embs=glove_embs)
    G_pos = Token_Dataset_nx(pos_corpus_toks, pos_vocab, dataset_name=pos_path[:-4])

    ## Create G-:
    neg_vocab, neg_corpus_toks, neg_corpus_strs, neg_oov_high_freqs =\
        get_vocab_data(neg_path, name='_neg', glove_embs=glove_embs)
    G_neg = Token_Dataset_nx(neg_corpus_toks, neg_vocab, dataset_name=neg_path[:-4])

    ## Find common nodes (C):
    C = set(pos_vocab['str2idx_map']).intersection(set(neg_vocab['str2idx_map']))
    logger.info(f"Intersection length of pos and neg |C|: {len(C)}")

    if limit_dataset is not None:
        C = set(list(C)[:limit_dataset])
        logger.debug(f"Resetting |C|: {limit_dataset}")

    extra_tokens = C - set(joint_vocab["str2idx_map"].keys())
    assert len(extra_tokens) >= 0, f'C has {len(extra_tokens)} extra tokens which are not in joint_vocab.'

    pos_neg_union = set(pos_vocab['str2idx_map']).union(
        set(neg_vocab['str2idx_map']))
    logger.info(f"Union length of pos and neg: {len(pos_neg_union)}")
    # assert pos_neg_union == set(joint_vocab['str2idx_map'].keys()),\
    #     f'{len(pos_neg_union)} == {len(joint_vocab["str2idx_map"].keys())} did not match.'
    logger.info(f"IoU percentage between POS and NEG: {(len(C) / len(pos_neg_union)) * 100}")

    # Check what portion of pos and neg vocab are actual english and OOV.
    # Need to remove OOV tokens from sentences?

    G_pos.add_edge_weights_pretrain()
    # pos_num_tokens = G_pos.num_tokens
    pos_node_list = list(G_pos.G.nodes)
    logger.info(f"Number of nodes {len(pos_node_list)} and edges {len(G_pos.G.edges)} in POS token graph")

    G_neg.add_edge_weights_pretrain()
    # neg_num_tokens = G_neg.num_tokens
    neg_node_list = list(G_neg.G.nodes)
    logger.info(f"Number of nodes {len(neg_node_list)} and edges {len(G_neg.G.edges)} in NEG token graph")

    dataset, ignored_tokens = get_pretrain_dataset(C, G, G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab)

    logger.warning(f'Total number of ignored tokens: {len(ignored_tokens)}')

    G.save_graph()

    return dataset, G, oov_embs, joint_vocab


def prepare_pretraining(model_type=cfg['pretrain']['model_type'],
                        oov_emb_filename=data_filename + '_OOV_vectors_dict',
                        graph_path=join(pretrain_dir, data_filename + '_token_nx.bin'),
                        vocab_path=join(pretrain_dir, data_filename + '_joint_vocab'),
                        dataset_path=join(pretrain_dir, data_filename + '_dataset')):
    if exists(dataset_path + '.json') and exists(join(
            pretrain_dir, oov_emb_filename + '.pkl')) and exists(graph_path):
        dataset = read_json(dataset_path, convert_ordereddict=False)
        joint_vocab = read_json(vocab_path)
        g = load_graph(graph_path)
        G = Token_Dataset_nx(None, None, None, None, None, G=g)
        oov_embs = load_pickle(filepath=pretrain_dir, filename=oov_emb_filename)
    else:
        dataset, G, oov_embs, joint_vocab = get_graph_and_dataset()

    adj, X = get_graph_inputs(G, oov_embs, joint_vocab, prepare_G=True)
    pretrain_dataloader = Pretrain_Dataset(dataset)

    logger.info(f'Pre-Training [{model_type}]')
    node_list = list(G.G.nodes)
    idx2str = joint_vocab['idx2str_map']
    if model_type == 'MLP':
        train_epochs_losses, state, save_path, X = mlp_trainer(
            X, pretrain_dataloader, in_dim=cfg['embeddings']['emb_dim'],
            hid_dim=cfg['gnn_params']['hid_dim'], epoch=cfg['pretrain']['epoch'],
            lr=cfg["pretrain"]["lr"], node_list=node_list, idx2str=idx2str)
    elif model_type == 'GCN':
        train_epochs_losses, state, save_path, X = gcn_trainer(
            adj, X, pretrain_dataloader, in_dim=cfg['embeddings']['emb_dim'],
            hid_dim=cfg['gnn_params']['hid_dim'], epoch=cfg['pretrain']['epoch'],
            lr=cfg["pretrain"]["lr"], node_list=node_list, idx2str=idx2str, model_type=model_type)
    else:
        raise NotImplementedError(f'[{model_type}] not found.')

    token2pretrained_embs = get_token2pretrained_embs(X, node_list, idx2str)

    return train_epochs_losses, state, save_path, joint_vocab, token2pretrained_embs, X


def get_pretrain_artifacts(
        epoch=cfg['pretrain']['epoch'], model_type=cfg['pretrain']['model_type'],
        vocab_path=join(pretrain_dir, data_filename + '_joint_vocab'),
        token_embs_path=join(pretrain_dir, 'pretrained', 'token2pretrained_'),
        pretrainedX_path=join(pretrain_dir, 'pretrained', 'X_')):
    token_embs_path = token_embs_path + str(epoch) + '.pt'
    pretrainedX_path = pretrainedX_path + str(epoch) + '.pt'

    state = None
    if exists(vocab_path + '.json') and exists(pretrainedX_path):
        logger.info(f'Loading Pretraining Artifacts from [{token_embs_path}] and [{vocab_path}]')
        vocab = read_json(vocab_path)
        X = load(pretrainedX_path)

        if exists(token_embs_path):
            token2pretrained_embs = load(token_embs_path)
        else:
            token2pretrained_embs = get_token2pretrained_embs(
                X, list(vocab['str2idx_map'].keys()), vocab['idx2str_map'])
    else:
        logger.info('Pretraining')
        _, state, _, vocab, token2pretrained_embs, X = prepare_pretraining(
            model_type=model_type)

    return state, vocab, token2pretrained_embs, X


def get_w2v_embs():
    joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
        get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs)

    X = train_w2v(joint_corpus_toks, list(joint_vocab['str2idx_map'].keys()),
                  in_dim=cfg['embeddings']['emb_dim'],
                  min_freq=cfg["prep_vecs"]["min_count"],
                  context=cfg["prep_vecs"]["window"])

    return joint_vocab['str2idx_map'], from_numpy(X)


if __name__ == "__main__":
    X, jv = get_w2v_embs()

    state, joint_vocab, token2pretrained_embs, X = get_pretrain_artifacts()
    C = set(glove_embs.keys()).intersection(set(token2pretrained_embs.keys()))
    logger.debug(f'Common vocab size: {len(C)}')
    words = ['nepal', 'italy', 'building', 'damage', 'kathmandu', 'water',
             'wifi', 'need', 'available', 'earthquake']
    X_glove = {word: glove_embs[word] for word in words}
    X_gcn = {word: token2pretrained_embs[word] for word in words}
    from Plotter.plot_functions import plot_vecs_color

    plot_vecs_color(tokens2vec=X_gcn, save_name='gcn_pretrained.pdf')
    plot_vecs_color(tokens2vec=X_glove, save_name='glove_pretrained.pdf')
    logger.debug(f'Word list: {words}')
