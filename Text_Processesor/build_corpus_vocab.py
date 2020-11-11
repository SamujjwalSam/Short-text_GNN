# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Generate token graph
__description__ : Details and usage.
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

from collections import Counter
from functools import partial
# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from Data_Handlers.torchtext_handler import prepare_fields, create_vocab,\
    create_tabular_dataset, dataset2iter, split_dataset
from Text_Processesor.tweet_normalizer import normalizeTweet
from config import configuration as cfg, platform as plat, username as user


def get_dataset_fields(
        csv_dir: str, csv_file: str, return_iter: bool = False,
        min_freq: int = 2, text_headers: list = ['text'], batch_size: int = 1,
        init_vocab: bool = True, labelled_data: bool = False,
        target_train_portion=None,
        embedding_dir: [None, str] = cfg["paths"]["embedding_dir"][plat][user],
        embedding_file: [None, str] = cfg["embeddings"]["embedding_file"],):
    ## Create tokenizer:
    tokenizer = partial(normalizeTweet, return_tokens=True)

    (TEXT, LABEL), labelled_fields, unlabelled_fields = prepare_fields(
        text_headers=text_headers, tokenizer=tokenizer)

    ## Create dataset from saved csv file:
    if labelled_data:
        dataset = create_tabular_dataset(csv_file, csv_dir, labelled_fields)
    else:
        dataset = create_tabular_dataset(csv_file, csv_dir, unlabelled_fields)

    if target_train_portion is not None:
        dataset, test = split_dataset(dataset, split_size=0.7)
        dataset, unused = split_dataset(dataset,
                                        split_size=target_train_portion)
        # if exists(join(csv_dir, csv_file + "_examples.pkl")):
        #     dataset = load_dataset(csv_dir, csv_file)
        # else:
        #     train, dataset = split_dataset(dataset, split_size=0.7)
        #     save_dataset(dataset, csv_dir, csv_file, fields=labelled_fields)

    ## Create vocabulary and mappings:
    if init_vocab:
        create_vocab(dataset, TEXT, LABEL, embedding_file=embedding_file,
                     embedding_dir=embedding_dir, min_freq=min_freq)
    if return_iter:
        iterator = dataset2iter(dataset, batch_size=batch_size)
        return dataset, (TEXT, LABEL), iterator

    if target_train_portion is not None:
        return dataset, (TEXT, LABEL), test
    return dataset, (TEXT, LABEL)


def build_corpus(df, txts: list = None, corpus: list = None):
    """Generates corpus (list of str) and vocab with occurrence count (dict of
     set of unique tokens).

    :param df:
    :param txts:
    :param corpus:
    :return:
    """
    if corpus is None:
        corpus = []

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.text)
    print(X.shape)
    print(vectorizer.get_feature_names())

    for txt in txts:
        # corpus = corpus + txt
        for token in txt:
            corpus.append(token)

    vocab_freq = Counter(corpus)

    return corpus, vocab_freq


class Vocabulary:
    """
    Taken from: https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks
    -python.html

    """

    UNK_token = 0  # Used for unknown tokens
    PAD_token = 1  # Used for padding short sentences
    SOS_token = 2  # Start-of-sentence token
    EOS_token = 3  # End-of-sentence token
    HASH_token = 4  # Used for #hashtags in tweets
    USER_token = 5  # Used for @user in tweets

    def __init__(self, name='vocab', tokenizer=None, stopwords: list = None):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {
            Vocabulary.UNK_token:  "<UNK>",
            Vocabulary.PAD_token:  "<PAD>",
            Vocabulary.SOS_token:  "SOS",
            Vocabulary.EOS_token:  "EOS",
            Vocabulary.HASH_token: "#HASH",
            Vocabulary.USER_token: "@USER",
        }
        self.examples = {}
        self.num_tokens = 6
        self.num_sentences = 0
        self.longest_sentence = 0
        self.shortest_sentence = 0
        if stopwords is None:
            stopwords = list(stop_words.ENGLISH_STOP_WORDS)
        if tokenizer is None:
            ## Create tokenizer:
            self.tokenizer = partial(normalizeTweet, return_tokens=True)

    def add_token(self, token):
        if token not in self.token2index:
            # First entry of token into vocabulary
            self.token2index[token] = self.num_tokens
            self.token2count[token] = 1
            self.index2token[self.num_tokens] = token
            self.num_tokens += 1
        else:
            # token exists; increase token count
            self.token2count[token] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        tokens = self.tokenizer(sentence)
        for token in tokens:
            sentence_len += 1
            self.add_token(token)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        if sentence_len < self.shortest_sentence:
            # This is the shortest sentence
            self.shortest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_token(self, index):
        return self.index2token[index]

    def to_index(self, token):
        return self.token2index[token]


def _test_Vocabulary():
    voc = Vocabulary('test')
    print(voc)
    corpus = ['This is the first sentence.',
              'This is the second.',
              'There is no sentence in this corpus longer than this one.',
              'My dog is named Patrick.']
    # print(corpus)
    for sent in corpus:
        voc.add_sentence(sent)

    print('Token 4 corresponds to token:', voc.to_token(4))
    print('Token "this" corresponds to index:', voc.to_index('this'))
    for token in range(voc.num_tokens):
        print(voc.to_token(token))

    sent_tkns = []
    sent_idxs = []
    for token in corpus[3].split(' '):
        sent_tkns.append(token)
        sent_idxs.append(voc.to_index(token))
    print(sent_tkns)
    print(sent_idxs)


if __name__ == "__main__":
    from tweet_normalizer import normalizeTweet

    t1 = "SC has first two presumptive cases of coronavirus, DHEC confirms "\
         "https://postandcourier.com/health/covid19/sc-has-first-two"\
         "-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae"\
         "-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source="\
         "twitter&utm_campaign=user-share… via @postandcourier"

    t2 = "#India dispatched 100,000 bottles of #RailNeer water 959-5085116"\
         " to quake-hit #Nepal on Saturday night. http://t.co/HXkVtw9hRo "\
         "#nepal via @oneindia"

    t1_toks = normalizeTweet(t1)
    t2_toks = normalizeTweet(t2)

    txts = [t1_toks, t2_toks]

    corpus1, vocab1 = build_corpus(txts)

    print(corpus1, vocab1)

    t3 = "#India dispatched 100,000 bottles of"
    t3_toks = normalizeTweet(t3)

    corpus3, vocab3 = build_corpus([t3_toks], corpus1)

    print(corpus3, vocab3)
    _test_Vocabulary()
