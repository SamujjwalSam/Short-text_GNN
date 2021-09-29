# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Trains static embeddings like Glove
__description__ : Useful for oov tokens
__project__     : Tweet_GNN_inductive
__classes__     :
__variables__   :
__methods__     :
__author__      : https://towardsdatascience.com/fine-tune-glove-embeddings
-using-mittens-89b5f3fe4c39
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:
"""

# import re
import csv
import numpy as np
# import pandas as pd
import pickle
from os.path import exists, join
from collections import Counter
from nltk.corpus import brown
from mittens import Mittens
from sklearn.feature_extraction.text import CountVectorizer

from File_Handlers.json_handler import read_json, save_json
from File_Handlers.pkl_handler import load_pickle, save_pickle
from config import configuration as cfg, platform as plat, username as user, dataset_dir
from Logger.logger import logger


def glove2dict(embedding_dir: str = cfg["paths"]["embedding_dir"][plat][user],
               embedding_file: str = cfg["embeddings"]["embedding_file"]) -> dict:
    """Loads Glove vectors and return dict.

    # get it from https://nlp.stanford.edu/projects/glove

    :param embedding_file:
    :param embedding_dir:
    :param glove_filename:
    :return:
    """
    glove_filename = join(embedding_dir, embedding_file + ".txt")

    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                 for line in reader}
    return embed


def preprocess_and_find_oov(datasets: tuple, common_vocab: dict = None,
                            glove_embs: dict = None, labelled_vocab_set: set = None,
                            special_tokens={'<unk>', '<pad>'}):
    """ Process and prepare data by removing stopwords, finding oovs and
     creating corpus.

    :param special_tokens:
    :param labelled_vocab_set: Tokens of labelled data with associate
    label vecs
    :param datasets:
    :param common_vocab: Vocab generated from all datasets
    :param oov_min_freq: Count of min freq oov token which should be removed
    :param glove_embs: Original glove embeddings in key:value format.
    :param glove_embs: glove embeddings

    :return:
    """
    if glove_embs is None:
        glove_embs = glove2dict()

    vocab_freq_set = set(common_vocab['freqs'].keys())
    vocab_s2i_set = set(common_vocab['str2idx_map'].keys())
    glove_set = set(glove_embs.keys())

    ## Tokens without embeddings:
    # oov = vocab_freq_set - glove_set
    # oov_freqs = {}
    # for token in oov:
    #     oov_freqs[token] = vocab['freqs'][token]

    ## Tokens with low freq in corpus, might be present in Glove:
    low = vocab_freq_set - vocab_s2i_set
    logger.info(f'Number of tokens with low freq in corpus (might be present '
                f'in Glove): [{len(low)}]')
    low_freqs = {}
    for token in low:
        low_freqs[token] = common_vocab['freqs'][token]

    ## Find OOV from labelled data:
    labelled_oov = labelled_vocab_set - vocab_s2i_set
    logger.info(f'Number of labelled OOV tokens (embedding needs to be'
                f' generated): [{len(labelled_oov)}]')

    ## Low freq tokens with embeddings; will be added back and used to create
    ## <unk> emb:
    low_glove = low.intersection(glove_set)
    logger.info(f'Number of low freq tokens with embeddings (will be '
                f'added back): [{len(low_glove)}]')

    ## Add 'low freq with embeddings' tokens back to vocab:
    start_idx = len(common_vocab['str2idx_map'])
    low_glove_freqs = {}
    for token in low_glove:
        common_vocab['str2idx_map'][token] = start_idx
        common_vocab['idx2str_map'].append(token)
        start_idx += 1

    ## Update vocab set after adding low freq tokens back:
    vocab_s2i_set.update(low_glove)

    ## Reinitialize low freq set after adding non-oov tokens back:
    ## Low freq tokens without embeddings:
    # low_oov = low - low_glove
    low_oov = vocab_freq_set - vocab_s2i_set
    logger.info(f'Number of low freq tokens without embeddings: '
                f'[{len(low_oov)}]')
    low_oov_freqs = {}
    for token in low_oov:
        low_oov_freqs[token] = common_vocab['freqs'][token]

    ## High freq but glove OOV except special tokens:
    high_oov = vocab_s2i_set - glove_set - special_tokens

    ## Add labelled oov tokens which does not have embedding:
    high_oov.update(labelled_oov - low_glove)

    logger.info(f'Number of high freq but OOV tokens: [{len(high_oov)}]')
    high_oov_freqs = {}
    for token in high_oov:
        high_oov_freqs[token] = common_vocab['freqs'][token]

    corpus = [[], []]
    corpus_toks = [[], []]
    for i, dataset in enumerate(datasets):
        for j, example in enumerate(dataset.examples):
            example_toks = []
            for token in example.text:
                try:  ## Ignore low freq OOV tokens:
                    low_oov_freqs[token]
                except KeyError:
                    example_toks.append(token)

            ## Ignore samples if no token left after cleaning:
            if len(example_toks) > 0:
                corpus[i].append(' '.join(example_toks))
                corpus_toks[i].append(example_toks)
            else:
                pass
                # logger.warning(f'Dataset [{i}] sample [{j}] has no token left'
                #                f' after cleaning: [{example.text}]')

    return high_oov_freqs, low_glove_freqs, corpus, corpus_toks


def preprocess_and_find_oov2(
        vocab: dict = None, glove_embs: dict = None, labelled_vocab_set: set = None,
        special_tokens={'<unk>', '<pad>'}, add_glove_tokens_back=True, limit_oov=5000):
    """ Process and prepare data by removing stopwords, finding oovs and creating corpus.

    :param add_glove_tokens_back: Adds low freq tokens which are present in glove
    :param special_tokens:
    :param labelled_vocab_set: Tokens of labelled data with associate
    label vecs
    :param datasets:
    :param vocab: Vocab generated from all datasets
    :param oov_min_freq: Count of min freq oov token which should be removed
    :param glove_embs: Original glove embeddings in key:value format.
    :param glove_embs: glove embeddings

    :return:
    """
    if glove_embs is None:
        glove_embs = glove2dict()

    vocab_freq_set = set(vocab['freqs'].keys())
    vocab_s2i_set = set(vocab['str2idx_map'].keys())
    glove_set = set(glove_embs.keys())

    ## Tokens without embeddings:
    # oov = vocab_freq_set - glove_set
    # oov_freqs = {}
    # for token in oov:
    #     oov_freqs[token] = vocab['freqs'][token]

    ## Tokens with low freq in corpus, might be present in Glove:
    low = vocab_freq_set - vocab_s2i_set
    logger.info(f'Number of tokens with low freq in corpus (might be present '
                f'in Glove): [{len(low)}]')
    low_freqs = {}
    for token in low:
        low_freqs[token] = vocab['freqs'][token]

    ## Find OOV from labelled data:
    labelled_oov = labelled_vocab_set - vocab_s2i_set
    logger.info(f'Number of labelled OOV tokens (embedding needs to be'
                f' generated): [{len(labelled_oov)}]')

    ## Low freq tokens with embeddings; will be added back and used to create
    ## <unk> emb:
    low_glove = low.intersection(glove_set)
    logger.info(f'Number of low freq tokens with embeddings (will be '
                f'added back): [{len(low_glove)}]')

    low_glove_freqs = {}
    if add_glove_tokens_back:
        ## Add 'low freq with embeddings' tokens back to vocab:
        start_idx = len(vocab['str2idx_map'])
        for token in low_glove:
            vocab['str2idx_map'][token] = start_idx
            vocab['idx2str_map'].append(token)
            low_glove_freqs[token] = vocab['freqs'][token]
            start_idx += 1

        ## Update vocab set after adding low freq tokens back:
        vocab_s2i_set.update(low_glove)

    ## Reinitialize low freq set after adding non-oov tokens back:
    ## Low freq tokens without embeddings:
    # low_oov = low - low_glove
    low_oov = vocab_freq_set - vocab_s2i_set
    logger.info(f'Number of low freq tokens without embeddings: '
                f'[{len(low_oov)}]')
    low_oov_freqs = {}
    for token in low_oov:
        low_oov_freqs[token] = vocab['freqs'][token]

    ## High freq but glove OOV except special tokens:
    high_oov = vocab_s2i_set - glove_set - special_tokens
    if limit_oov is not None and limit_oov < len(high_oov):
        logger.info(f'Limit OOV size to [{limit_oov}] from {len(high_oov)}')
        high_oov = set(list(high_oov)[:limit_oov])

    ## Add labelled oov tokens which does not have embedding:
    high_oov.update(labelled_oov - low_glove)

    logger.info(f'Number of high freq but OOV tokens: [{len(high_oov)}]')
    high_oov_freqs = {}
    for token in high_oov:
        high_oov_freqs[token] = vocab['freqs'][token]

    return high_oov_freqs, low_glove_freqs, low_oov_freqs


def create_clean_corpus(dataset, low_oov_ignored):
    """ Ignores low freq OOV tokens and creates corpus without those tokens.

    :param dataset: TorchText dataset
    :param low_oov_ignored: set of tokens to be ignored.
    :return:
        corpus_strs: str per example (list of str)
        corpus_toks: tokenized text per examples (list of list of str)
        ignored_examples: list of example id and text for which no token was left after preprocessing.
    """
    corpus_strs = []
    corpus_toks = []
    ignored_examples = []
    ## Creates list of tokens after removing low freq OOV tokens:
    for j, example in enumerate(dataset.examples):
        example_toks = []
        for token in example.text:
            ## Ignore low freq OOV tokens:
            if token not in low_oov_ignored:
                example_toks.append(token)

        ## Ignore samples if no token left after cleaning:
        if len(example_toks) > 0:
            corpus_strs.append(' '.join(example_toks))
            corpus_toks.append(example_toks)
        else:
            ignored_examples.append((j, example.text))

    logger.warning(f'{len(ignored_examples)} examples are ignored as no token'
                   f' was left after cleaning.')

    return corpus_strs, corpus_toks, ignored_examples


def process_data(data: list, glove_embs: dict = None,
                 # stopwords: list = stop_words.ENGLISH_STOP_WORDS,
                 remove_rare_oov: bool = False):
    """ Process and prepare data by removing stopwords, finding oovs and
     creating corpus.

    :param glove_embs: Original glove embeddings in key:value format.
    :param data: List[list]: List of tokenized sentences (list of token
     str list).
    :param stopwords:
    :param glove_path: Full path to glove file.
    :param remove_rare_oov: Boolean to signify if tokens with very less freq
     should be removed?

    :return:
    """
    if glove_embs is None:
        glove_embs = glove2dict()

    # nonstop_tokens = [token for token in data if (token not in stopwords)]
    nonstop_tokens = [token for token in data]

    ## Tokens (repeated) not present in glove:
    oov = [token for token in nonstop_tokens if token not in glove_embs.keys()]

    ## Unique oov tokens
    oov_vocabs = list(set(oov))

    ## List of all the tokens in a str within a list as a large document
    doc = [' '.join(nonstop_tokens)]

    ## Remove rare oov words to reduce vocab size
    if remove_rare_oov:
        oov_rare = get_rareoov(oov, 1)
        oov_vocabs = list(set(oov) - set(oov_rare))
        tokens = [token for token in nonstop_tokens if token not in
                  oov_rare]
        doc = [' '.join(tokens)]

    return oov_vocabs, doc


def get_rareoov(xdict, val):
    return [k for (k, v) in Counter(xdict).items() if v <= val]


def calculate_cooccurrence_mat(oov_vocab: list, corpus_str: list):
    """ Calculates token co-occurrence matrix for oov tokens.

    :param oov_vocab:
    :param corpus_str:
    :return:
    """
    ## Get oov token freq in corpus:
    cv = CountVectorizer(ngram_range=(1, 1), vocabulary=oov_vocab)
    X = cv.fit_transform(corpus_str)

    ## X.T * X converts doc-token matrix to token-token matrix:
    Xc = (X.T * X)
    Xc.setdiag(0)
    coocc_ar = Xc.toarray()
    return coocc_ar


def train_mittens(coocc_ar, oov_vocabs, pre_glove, emb_dim=cfg['embeddings'][
    'emb_dim'], max_iter=300,
                  glove_oov_save_path=None, dataset_dir=dataset_dir,
                  embedding_file=cfg["embeddings"]["embedding_file"],
                  dataset_name=cfg['data']['name']):
    """

    :param coocc_ar:
    :param oov_vocabs:
    :param pre_glove:
    :param emb_dim:
    :param max_iter:
    :param glove_oov_save_path:
    :param dataset_dir:
    :param embedding_file:
    :param dataset_name:
    :return:
    """
    mittens_model = Mittens(n=emb_dim, max_iter=max_iter)

    new_embeddings = mittens_model.fit(
        coocc_ar,
        vocab=oov_vocabs,
        initial_embedding_dict=pre_glove)

    newglove = dict(zip(oov_vocabs, new_embeddings))
    if glove_oov_save_path is None:
        glove_oov_save_path = join(dataset_dir, embedding_file + dataset_name +
                                   '_oov.pkl')
    f = open(glove_oov_save_path, "wb")
    pickle.dump(newglove, f)
    f.close()

    return newglove


def get_oov_tokens(dataset, dataname, data_dir, vocab, glove_embs):
    datapath = join(data_dir, dataname)
    if exists(datapath + '_high_oov.json')\
            and exists(datapath + '_corpus.json')\
            and exists(datapath + '_corpus_toks.json'):
        high_oov = read_json(datapath + '_high_oov')
        low_glove = read_json(datapath + '_low_glove')
        low_oov = read_json(datapath + '_low_oov')
        corpus = read_json(datapath + '_corpus', convert_ordereddict=False)
        corpus_toks = read_json(datapath + '_corpus_toks', convert_ordereddict=False)
        # vocab = read_json(datapath + 'vocab', convert_ordereddict=False)
    else:
        ## Get all OOVs which does not have Glove embedding:
        high_oov, low_glove, low_oov = preprocess_and_find_oov2(vocab, glove_embs=glove_embs,
                                                                labelled_vocab_set=set(vocab['str2idx_map'].keys()))

        corpus, corpus_toks, _ = create_clean_corpus(dataset, low_oov)

        ## Save token sets: high_oov, low_glove, corpus, corpus_toks
        save_json(high_oov, datapath + '_high_oov')
        save_json(low_glove, datapath + '_low_glove')
        save_json(low_oov, datapath + '_low_oov')
        save_json(corpus, datapath + '_corpus')
        save_json(corpus_toks, datapath + '_corpus_toks')
        # save_json(vocab, datapath + 'vocab', overwrite=True)

    return high_oov, low_glove, low_oov, corpus, corpus_toks


def get_oov_vecs(high_oov_tokens, corpus, dataname, data_dir, glove_embs,
                 mittens_iter=cfg['model']['mittens_iter']):
    logger.info(f'Get embeddings for OOV tokens')
    oov_filename = dataname + '_OOV_vectors_dict'  # + str(mittens_iter)
    if exists(join(data_dir, oov_filename + '.pkl')):
        logger.info('Read OOV embeddings:')
        oov_embs = load_pickle(filepath=data_dir, filename=oov_filename)
    else:
        logger.info('Create OOV embeddings using Mittens:')
        # high_oov, low_glove, low_oov, corpus, corpus_toks = \
        #     get_oov_tokens(dataset, dataname, data_dir, vocab, glove_embs)
        # high_oov_tokens_list = list(high_oov.keys())
        oov_mat_coo = calculate_cooccurrence_mat(high_oov_tokens, corpus)
        oov_embs = train_mittens(oov_mat_coo, high_oov_tokens, glove_embs, max_iter=mittens_iter)
        save_pickle(oov_embs, filepath=data_dir, filename=oov_filename, overwrite=True)

    return oov_embs


def main():
    glove_embs = glove2dict()
    oov_vocabs, corpus = process_data(brown.words()[:2000],
                                      glove_embs=glove_embs)
    coo_mat = calculate_cooccurrence_mat(oov_vocabs, corpus)
    new_glove_embs = train_mittens(coo_mat, oov_vocabs, glove_embs)
    return new_glove_embs


if __name__ == "__main__":
    main()
