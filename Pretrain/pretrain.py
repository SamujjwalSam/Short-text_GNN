# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Contrastive pretraining script
__description__ : Details and usage.
__project__     : GCPD
__classes__     : GCPD
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
from functools import partial
from scipy.special import softmax
from torch import cuda, save, load, manual_seed, backends, from_numpy
from torch.utils.data import Dataset
from networkx import adjacency_matrix
from os import environ
from os.path import join, exists
from sklearn.preprocessing import MultiLabelBinarizer

from Text_Encoder.TextEncoder import train_w2v, load_word2vec
from Pretrainer.mlp_trainer import mlp_trainer
from Pretrainer.gcn_trainer import gcn_trainer
# from Trainer.lstm_trainer import LSTM_trainer
from Utils.utils import load_graph, sp_coo2torch_coo, get_token2pretrained_embs
    # save_token2pretrained_embs, load_token2pretrained_embs
from File_Handlers.csv_handler import read_csv
from File_Handlers.json_handler import save_json, read_json
from File_Handlers.pkl_handler import save_pickle, load_pickle
# from Text_Processesor.tokenizer import BERT_tokenizer
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Text_Processesor.tweet_normalizer import normalizeTweet
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Text_Encoder.finetune_static_embeddings import glove2dict, train_mittens,\
    calculate_cooccurrence_mat, preprocess_and_find_oov2, create_clean_corpus
from config import configuration as cfg, platform as plat, username as user, \
    pretrain_dir, emb_dir
from Logger.logger import logger

if cuda.is_available():
    environ["CUDA_VISIBLE_DEVICES"] = str(cfg['cuda']['cuda_devices'][plat][user])
    cuda.set_device(cfg['cuda']['cuda_devices'][plat][user])


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


def get_xclusive_tokens(pos_vocab, neg_vocab, min_ratio=.9):
    pos_s2i = set(pos_vocab['str2idx_map']) - {'<unk>', '<pad>'}
    neg_s2i = set(neg_vocab['str2idx_map']) - {'<unk>', '<pad>'}
    # pos_freq = set(pos_vocab['freqs'])
    # neg_freq = set(neg_vocab['freqs'])
    Epos = pos_s2i - neg_s2i
    Eneg = neg_s2i - pos_s2i
    logger.info(f'Exclusive pos {len(Epos)} and neg {len(Eneg)} counts')
    nonX_toks = pos_s2i.union(neg_s2i)
    nonX_toks = nonX_toks - Epos.union(Eneg)
    # logger.info(f'Exclusive pos {len(Epos)} and neg {len(Eneg)} tokens')
    pos_ratio_toks = []
    neg_ratio_toks = []
    for tok in nonX_toks:
        total_freq = pos_vocab['freqs'][tok] + neg_vocab['freqs'][tok]
        if tok in neg_s2i:
            pos_ratio = pos_vocab['freqs'][tok] / total_freq
            if pos_ratio >= min_ratio:
                pos_ratio_toks.append(tok)
        if tok in pos_s2i:
            neg_ratio = neg_vocab['freqs'][tok] / total_freq
            if neg_ratio >= min_ratio:
                neg_ratio_toks.append(tok)

    Epos.update(pos_ratio_toks)
    Eneg.update(neg_ratio_toks)
    logger.debug(f'Added back {len(pos_ratio_toks)} pos and '
                 f'{len(neg_ratio_toks)} neg tokens to Exclusive sets')
    logger.info(f'Total pos {len(Epos)} and neg {len(Eneg)} tokens selected')

    return Epos, Eneg


def get_pretrain_dataset(G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab,
                         limit_dataset=None):
    """ Creates the dataset for pretraining.

     Fetches N+(c) and N-(c) for each common token between pos and neg graphs.

    :param G:
    :param G_pos:
    :param G_neg:
    :param joint_vocab:
    :param pos_vocab:
    :param neg_vocab:
    :param limit_dataset:
    :return:
    """
    ## Find common nodes (C):
    C = set(pos_vocab['str2idx_map']).intersection(set(neg_vocab['str2idx_map']))
    logger.info(f"Intersection length of pos and neg |C|: {len(C)}")

    # logger.debug(f'Find exclusive pos and neg tokens')
    # Cpos, Cneg = get_xclusive_tokens(pos_vocab, neg_vocab)

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
        # N_pos_txt = set(N_pos_txt)
        # N_neg_txt = set(N_neg_txt)
        # overlap_txt = N_pos_txt.intersection(N_neg_txt)
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


def get_sel_samples(C, G, vocab, joint_vocab, k=20):
    data = {}
    ignored_tokens = set()
    for i, token in enumerate(C):
        N_id = []
        N_id_wt = []
        N_txt = []
        if token not in vocab['str2idx_map']:
            logger.debug(token)
        for n in G.G.neighbors(vocab['str2idx_map'][token]):
            try:
                n_txt = vocab['idx2str_map'][n]
            except IndexError as e:
                logger.debug((token, n))
                continue
            n_id = joint_vocab['str2idx_map'][n_txt]
            if n_id in N_id:
                continue
            ## Take freq values from combined graph:
            # wt = global_G.G[joint_vocab['str2idx_map'][token]][n_id]
            wt = G.G[vocab['str2idx_map'][token]][vocab['str2idx_map'][n_txt]]
            N_id.append(n_id)
            N_id_wt.append(wt['freq'])
            N_txt.append(n_txt)

        if len(N_id) == 0:
            # logger.warning(f'Token {token} has 0 POS (+) neighbors.')
            ignored_tokens.add(token)
            continue
        ## Sample positive neighbors based on freq:
        # N_sel = random.choices(N_id, N_id_wt, k=k)
        N_id_wt = np.power(N_id_wt, (3 / 4)).tolist()
        # N_id_wt2 = np.log(N_id_wt).tolist()
        probas = softmax(N_id_wt)
        probas_nnz = np.count_nonzero(probas)
        # logger.debug(probas_nnz)
        if probas_nnz < min(k, len(N_id)):
            logger.warn(f'Fewer non-zero entries in p {probas_nnz} than size {min(k, len(N_id))}')
            N_id_wt = np.log(N_id_wt).tolist()
            probas = softmax(N_id_wt)
            # probas_nnz = np.count_nonzero(probas)
        N_sel = np.random.choice(N_id, size=min(k, len(N_id)), replace=False, p=probas).tolist()

        # N_frq, N_wt = [], []
        # for n in N_sel:
        #     N_frq = N_wt_dict[n]['freq']
        #     N_wt = N_wt_dict[n]['weight']
        data[joint_vocab['str2idx_map'][token]] = N_sel

    return data, ignored_tokens


def get_new_pretrain_dataset(G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab,
                             limit_dataset=None, N=40):
    """ Creates the dataset for pretraining.

    Fetches exclusive pos and neg token set which are present either in pos and neg graph with high freq.

    :param limit_dataset:
    :param N: Number of total neighbors to select for each example token
    :param G:
    :param G_pos:
    :param G_neg:
    :param joint_vocab:
    :param pos_vocab:
    :param neg_vocab:
    :return:
    """
    logger.debug(f'Find exclusive pos and neg tokens')
    Cpos, Cneg = get_xclusive_tokens(pos_vocab, neg_vocab)

    if limit_dataset is not None:
        Cpos = set(list(Cpos)[:limit_dataset // 2])
        logger.debug(f"Resetting |Cpos|: {len(Cpos)}")
        Cneg = set(list(Cneg)[:limit_dataset // 2])
        logger.debug(f"Resetting |Cneg|: {len(Cneg)}")

    # extra_tokens = C - set(joint_vocab["str2idx_map"].keys())
    # assert len(extra_tokens) >= 0, f'C has {len(extra_tokens)} extra tokens which are not in joint_vocab.'

    pos_neg_union = set(pos_vocab['str2idx_map']).union(
        set(neg_vocab['str2idx_map']))
    logger.info(f"Union length of pos and neg: {len(pos_neg_union)}")
    # assert pos_neg_union == set(joint_vocab['str2idx_map'].keys()),\
    #     f'{len(pos_neg_union)} == {len(joint_vocab["str2idx_map"].keys())} did not match.'
    logger.info(f"Overlap between selected and total vocab: {(len(Cpos.union(Cneg)) / len(pos_neg_union)) * 100}")

    dataset = []
    pos_data, pos_ignored = get_sel_samples(Cpos, G_pos, pos_vocab, joint_vocab)
    neg_data, neg_ignored = get_sel_samples(Cneg, G_neg, neg_vocab, joint_vocab)

    for node_id in pos_data:
        N_pos = pos_data[node_id]
        N_neg = random.sample(list(neg_data), k=(N - len(N_pos)))
        dataset.append((node_id, N_pos, N_neg))

    for node_id in neg_data:
        N_neg = neg_data[node_id]
        N_pos = random.sample(list(pos_data), k=(N - len(N_neg)))
        dataset.append((node_id, N_neg, N_pos))

    logger.critical(
        f'Pretraining dataset size: {len(dataset)}; portion of total vocab: '
        f'{(len(dataset) / len(joint_vocab["idx2str_map"])):2.4}')
    return dataset, pos_ignored.update(neg_ignored)


def get_vocab_data(path=join(pretrain_dir, data_filename + "_multihot.csv"),
                   name='_joint', glove_embs=None, min_freq=cfg['pretrain']['min_freq'], read_input=False):
    """ Creates cleaned corpus and finds oov tokens.

    :param read_input:
    :param path:
    :param name:
    :param glove_embs:
    :param min_freq:
    :return:
    """
    if exists(join(pretrain_dir, data_filename + name + '_corpus_toks.json'))\
            and exists(join(pretrain_dir, data_filename + name + '_vocab.json')):
        vocab = read_json(join(pretrain_dir, data_filename + name + '_vocab'))
        oov_high_freqs = read_json(join(pretrain_dir, data_filename + name + '_oov_high_freqs'))
        corpus_toks = read_json(join(pretrain_dir, data_filename + name + '_corpus_toks'),
                                convert_ordereddict=False)
        corpus_strs = read_json(join(pretrain_dir, data_filename + name + '_corpus_strs'),
                                convert_ordereddict=False)
    else:
        if read_input:
            read_input_files()
        # Create tokenizer:
        logger.info(f'Using normalizeTweet tokenizer')
        tokenizer = partial(normalizeTweet, return_tokens=True)

        dataset, (fields, LABEL) = get_dataset_fields(
            csv_dir='', csv_file=path, min_freq=min_freq, tokenizer=tokenizer)

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


def get_graph_inputs(G, oov_embs, joint_vocab, G_node_list=None, glove_embs=None, get_adj=True):
    logger.info('Accessing graph node embeddings:')
    emb_filename = join(pretrain_dir, data_filename + "_emb.pt")
    if exists(emb_filename):
        X = load(emb_filename)
    else:
        logger.info('Get node embeddings from graph')
        if glove_embs is None:
            glove_embs = glove2dict()
        X = G.get_node_embeddings_from_dict(oov_embs, glove_embs, joint_vocab['idx2str_map'])
        # X = sp_coo2torch_coo(X)
        save(X, emb_filename)

    logger.info('Accessing adjacency matrix')

    if G_node_list is None:
        G_node_list = list(G.G.nodes)

    # adj = None
    ## Create adjacency matrix if required:
    if get_adj:
        adj = adjacency_matrix(G.G, nodelist=G_node_list, weight='weight')
        adj = sp_coo2torch_coo(adj)
        logger.info('Normalize token graph:')
        adj = G.normalize_adj(adj)

        return X, adj
    return X


def get_graph_and_dataset(glove_embs=None, limit_dataset=None):
    # df, joint_path, pos_path, neg_path = read_dataset(pretrain_dir, data_filename)
    if glove_embs is None:
        glove_embs = glove2dict()
    # oov_emb_filename = data_filename + '_OOV_vectors_dict'
    oov_emb_filename = 'disaster_binary_pretrain_OOV_vectors_dict'
    joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
        get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs, read_input=True)
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
    logger.info(f"Number of nodes {len(G_node_list)} and edges {len(G.G.edges)} in joint token graph")
    # token2label_vec_map = freq_tokens_per_class(df, normalize=False)

    ## Create G+:
    pos_vocab, pos_corpus_toks, pos_corpus_strs, pos_oov_high_freqs =\
        get_vocab_data(pos_path, name='_pos', glove_embs=glove_embs)
    G_pos = Token_Dataset_nx(pos_corpus_toks, pos_vocab, dataset_name=pos_path[:-4])

    ## Create G-:
    neg_vocab, neg_corpus_toks, neg_corpus_strs, neg_oov_high_freqs =\
        get_vocab_data(neg_path, name='_neg', glove_embs=glove_embs)
    G_neg = Token_Dataset_nx(neg_corpus_toks, neg_vocab, dataset_name=neg_path[:-4])

    # ## Find common nodes (C):
    # C = set(pos_vocab['str2idx_map']).intersection(set(neg_vocab['str2idx_map']))
    # logger.info(f"Intersection length of pos and neg |C|: {len(C)}")
    #
    # if limit_dataset is not None:
    #     C = set(list(C)[:limit_dataset])
    #     logger.debug(f"Resetting |C|: {limit_dataset}")
    #
    # extra_tokens = C - set(joint_vocab["str2idx_map"].keys())
    # assert len(extra_tokens) >= 0, f'C has {len(extra_tokens)} extra tokens which are not in joint_vocab.'
    #
    # pos_neg_union = set(pos_vocab['str2idx_map']).union(
    #     set(neg_vocab['str2idx_map']))
    # logger.info(f"Union length of pos and neg: {len(pos_neg_union)}")
    # # assert pos_neg_union == set(joint_vocab['str2idx_map'].keys()),\
    # #     f'{len(pos_neg_union)} == {len(joint_vocab["str2idx_map"].keys())} did not match.'
    # logger.info(f"IoU percentage between POS and NEG: {(len(C) / len(pos_neg_union)) * 100}")

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

    # dataset, ignored_tokens = get_pretrain_dataset(G, G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab)
    dataset, ignored_tokens = get_new_pretrain_dataset(G_pos, G_neg, joint_vocab, pos_vocab, neg_vocab)

    if ignored_tokens is not None:
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

    X, adj = get_graph_inputs(G, oov_embs, joint_vocab, get_adj=True)
    pretrain_dataloader = Pretrain_Dataset(dataset)

    logger.info(f"Pre-Training [{model_type}] for {cfg['pretrain']['epoch']} epochs")
    node_list = list(G.G.nodes)
    idx2str = joint_vocab['idx2str_map']
    if model_type == 'MLP':
        train_epochs_losses, X = mlp_trainer(
            X, pretrain_dataloader, in_dim=cfg['embeddings']['emb_dim'],
            hid_dim=cfg['gnn_params']['hid_dim'], epoch=cfg['pretrain']['epoch'],
            lr=cfg["pretrain"]["lr"], node_list=node_list, idx2str=idx2str)
    elif model_type == 'GCN':
        train_epochs_losses, X = gcn_trainer(
            adj, X, pretrain_dataloader, in_dim=cfg['embeddings']['emb_dim'],
            hid_dim=cfg['gnn_params']['hid_dim'], epoch=cfg['pretrain']['epoch'],
            lr=cfg["pretrain"]["lr"], node_list=node_list, idx2str=idx2str, model_type=model_type)
    else:
        raise NotImplementedError(f'[{model_type}] not found.')

    token2pretrained_embs = get_token2pretrained_embs(X, node_list, idx2str)

    return train_epochs_losses, joint_vocab, token2pretrained_embs, X


def get_pretrain_artifacts(
        epoch=cfg['pretrain']['epoch'], model_type=cfg['pretrain']['model_type'],
        vocab_path=join(pretrain_dir, data_filename + '_joint_vocab'),
        token_embs_path=join(pretrain_dir, 'pretrained', 'token2pretrained_'),
        pretrainedX_path=join(pretrain_dir, 'pretrained', 'X_')):
    token_embs_path = token_embs_path + str(epoch) + '.pt'
    pretrainedX_path = pretrainedX_path + str(epoch) + '.pt'

    ## Use this:
    # load_token2pretrained_embs()

    if exists(pretrainedX_path):
        logger.info(f'Loading Pretraining Artifacts from [{token_embs_path}]')
        if not exists(vocab_path + '.json'):
            _, _, _, _ = get_graph_and_dataset()
            logger.info(f'[{vocab_path}] NOT found; Generating...')
        vocab = read_json(vocab_path)

        X = load(pretrainedX_path)

        if exists(token_embs_path):
            token2pretrained_embs = load(token_embs_path)
        else:
            token2pretrained_embs = get_token2pretrained_embs(
                X, list(vocab['str2idx_map'].keys()), vocab['idx2str_map'])
    else:
        logger.fatal(f'Pretrained [{vocab_path + ".json"}] or [{pretrainedX_path}] NOT found.')
        _, vocab, token2pretrained_embs, X = prepare_pretraining(
            model_type=model_type)

    return vocab, token2pretrained_embs, X


def get_w2v_embs(glove_embs):
    joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
        get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs, read_input=True)

    X = train_w2v(joint_corpus_toks, list(joint_vocab['str2idx_map'].keys()),
                  in_dim=cfg['embeddings']['emb_dim'],
                  min_freq=cfg['pretrain']['min_freq'],
                  context=cfg["prep_vecs"]["window"])

    return joint_vocab['str2idx_map'], from_numpy(X)


# def get_crisisNLP_embs():
#     if exists(join(emb_dir, 'crisisNLP_0.pt')) and exists(join(emb_dir, 'str2idx.pt')):
#         X, _ = load_token2pretrained_embs(
#             '1', pretrainedX_path=join(emb_dir, '', 'crisisNLP_'),
#             token2pretrained_path=join(emb_dir, '', 'crisisNLP_token2pretrained_'))
#         str2idx = read_json(join(emb_dir, 'str2idx.pt'))
#     else:
#         joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
#             get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs)
#
#         crisisnlp_model = load_word2vec(model_file_name='crisisNLP_word_vector_w2v.bin')
#         ordered_tokens = list(joint_vocab['str2idx_map'].keys())
#         X = [np.array(
#             [crisisnlp_model[w] if w in crisisnlp_model else np.random.uniform(
#                 -0.25, 0.25, crisisnlp_model.vector_size) for w in ordered_tokens])][0]
#
#         X = from_numpy(X)
#
#         save_token2pretrained_embs(
#             X, ordered_tokens, joint_vocab['idx2str_map'], '0',
#             pretrainedX_path=join(emb_dir, '', 'crisisNLP_'),
#             token2pretrained_path=join(emb_dir, '', 'crisisNLP_token2pretrained_'))
#         str2idx = joint_vocab['str2idx_map']
#
#     return str2idx, X


def get_cnlp_embs(glove_embs):
    joint_vocab, joint_corpus_toks, joint_corpus_strs, joint_oov_high_freqs =\
        get_vocab_data(joint_path, name='_joint', glove_embs=glove_embs, read_input=True)

    crisisnlp_model = load_word2vec(model_file_name='crisisNLP_word_vector_w2v.bin')
    ordered_tokens = list(joint_vocab['str2idx_map'].keys())
    X = [np.array(
        [crisisnlp_model[w] if w in crisisnlp_model else np.random.uniform(
            -0.25, 0.25, crisisnlp_model.vector_size) for w in ordered_tokens])][0]

    X = from_numpy(X)

    return joint_vocab['str2idx_map'], X


if __name__ == "__main__":
    # jv, X = get_crisisNLP_embs()
    # jv, X = get_w2v_embs()
    # from Plotter.plot_functions import test_plot_heatmap
    #
    # test_plot_heatmap()

    joint_vocab, token2pretrained_embs, X = get_pretrain_artifacts()
    C = set(glove_embs.keys()).intersection(set(token2pretrained_embs.keys()))
    logger.debug(f'Common vocab size: {len(C)}')
    words = ['nepal', 'queensland', 'building', 'damage', 'kathmandu', 'water',
             'wifi', 'need', 'available', 'earthquake']

    common = ['common', 'works', 'fear', 'system', 'honestly', 'such', 'trapped', 'technology', 'collect', 'thoughts',
              'rise', 'hours', 'dollars']
    words.extend(common)

    pos_exp = ['linking', 'corporation', 'unclear', 'filling', 'additional', 'pledges', 'slide', 'reversing',
               'particularly', 'cared', 'miraculously', 'properly', 'recovering']
    words.extend(pos_exp)

    neg_exp = ['successful', 'sounding', 'equally', 'antique', 'beautifully', 'stink', 'sauce', 'homecoming',
               'emotions', 'lick', 'atheist', 'fancy', 'coconut']
    words.extend(neg_exp)

    X_glove = {word: glove_embs[word] for word in words}
    X_gcn = {word: token2pretrained_embs[word] for word in words}
    from Plotter.plot_functions import plot_vecs_color

    plot_vecs_color(tokens2vec=X_gcn, save_name='gcn_pretrained.pdf')
    plot_vecs_color(tokens2vec=X_glove, save_name='glove_pretrained.pdf')
    logger.debug(f'Word list: {words}')
