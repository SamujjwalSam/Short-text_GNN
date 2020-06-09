# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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

import csv
import numpy as np
import pandas as pd
import pickle
from os.path import join
from collections import Counter
from nltk.corpus import brown
from mittens import Mittens
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

from config import configuration as cfg, platform as plat, username as user
from Logger.logger import logger


def glove2dict(embedding_dir=cfg["paths"]["embedding_dir"][plat][user],
               embedding_file=cfg["embeddings"]["embedding_file"]):
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


def preprocess_and_find_oov(datasets: tuple, common_vocab=None,
                            glove_embs: dict = None, stopwords: list = None,
                            oov_min_freq: int = 2,
                            ):
    """ Process and prepare data by removing stopwords, finding oovs and
     creating corpus.

    :param datasets:
    :param common_vocab: Vocab generated from all datasets
    :param oov_min_freq: Count of min freq oov token which should be removed
    :param glove_embs: Original glove embeddings in key:value format.
    :param stopwords:
    :param glove_embs: glove embeddings

    :return:
    """
    ## TODO: Set STOPWORDs' freq = 0 to ignore them.
    if stopwords is None:
        stopwords = list(stop_words.ENGLISH_STOP_WORDS)
    if glove_embs is None:
        glove_embs = glove2dict()

    vocab_freq_set = set(common_vocab['freqs'].keys())
    glove_set = set(glove_embs.keys())
    vocab_s2i_set = set(common_vocab['str2idx_map'].keys())

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

    ## Low freq tokens with embeddings; will be used to create <unk> emb:
    low_glove = low.intersection(glove_set)
    logger.info(f'Number of low freq tokens with embeddings (should be '
                f'added back): [{len(low_glove)}]')
    ## Add these tokens back to vocab
    start_idx = len(common_vocab['str2idx_map'])
    low_glove_freqs = {}
    for token in low_glove:
        common_vocab['str2idx_map'][token] = start_idx
        common_vocab['idx2str_list'].append(token)
        start_idx += 1

    ## Reinitialize low freq set after adding non-oov tokens back:
    ## Low freq tokens without embeddings:
    # low_oov = low - low_glove
    low_oov = vocab_freq_set - set(common_vocab['str2idx_map'].keys())
    logger.info(f'Number of low freq tokens without embeddings: '
                f'[{len(low_oov)}]')
    low_oov_freqs = {}
    for token in low_oov:
        low_oov_freqs[token] = common_vocab['freqs'][token]

    ## Not low freq but glove OOV:
    high_oov = vocab_s2i_set - glove_set
    logger.info(f'Number of high freq but glove OOV tokens:'
                f' [{len(high_oov)}]')
    high_oov_freqs = {}
    for token in high_oov:
        high_oov_freqs[token] = common_vocab['freqs'][token]

    ## Low freq tokens without embeddings (subset of oov):
    # low_freq_oov = low_freqs - low_freq_glove

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
                logger.warning(f'Dataset [{i}] sample [{j}] has no token left'
                               f' after cleaning: [{example.text}]')

    return high_oov_freqs, low_glove_freqs, corpus, corpus_toks


def process_data(data: list, glove_embs: dict = None,
                 stopwords: list = stop_words.ENGLISH_STOP_WORDS,
                 remove_rare_oov: bool = False):
    """ Process and prepare data by removing stopwords, finding oovs and
     creating corpus.

    :param glove_embs: Original glove embeddings in key:value format.
    :param data: List[List[str]]: List of tokenized sentences (list of token
     str list[str]).
    :param stopwords:
    :param glove_path: Full path to glove file.
    :param remove_rare_oov: Boolean to signify if tokens with very less freq
     should be removed?

    :return:
    """
    ## Tokens other than stopwords:
    if stopwords is None:
        stopwords = list(stop_words.ENGLISH_STOP_WORDS)
    if glove_embs is None:
        glove_embs = glove2dict()

    nonstop_tokens = [token for token in data if (token not in
                                                          stopwords)]

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


def train_model(coocc_ar, oov_vocabs, pre_glove, emb_dim=100, max_iter=1000,
                glove_oov_save_path=None,
                dataset_dir=cfg["paths"]["dataset_dir"][plat][user],
                embedding_file=cfg["embeddings"]["embedding_file"],
                dataset_name=cfg["data"]["source"]['labelled']
                             + cfg["data"]["target"]['labelled']):
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


def main():
    glove_embs = glove2dict()
    oov_vocabs, corpus = process_data(brown.words()[:2000],
                                      glove_embs=glove_embs)
    coo_mat = calculate_cooccurrence_mat(oov_vocabs, corpus)
    new_glove_embs = train_model(coo_mat, oov_vocabs, glove_embs)
    return new_glove_embs


if __name__ == "__main__":
    main()
