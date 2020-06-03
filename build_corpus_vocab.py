# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
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
from sklearn.feature_extraction.text import CountVectorizer

from tweet_normalizer import normalizeTweet
from Data_Handlers.torchtext_handler import prepare_fields, create_vocab,\
    create_tabular_dataset, dataset2iter


def torchtext_corpus(csv_dir, csv_file, embedding_file=None,
                     embedding_dir=None, return_iter=False,
                     text_headers=['text'], batch_size=1):
    ## Create tokenizer:
    tokenizer = partial(normalizeTweet, return_tokens=True)

    (TEXT, LABEL), labelled_fields, unlabelled_fields = prepare_fields(
        text_headers=text_headers, tokenizer=tokenizer)

    ## Create dataset from saved csv file:
    dataset = create_tabular_dataset(csv_file, csv_dir, unlabelled_fields)

    ## Create vocabulary and mappings:
    create_vocab(dataset, TEXT, embedding_file=embedding_file,
                 embedding_dir=embedding_dir)
    if return_iter:
        iterator = dataset2iter(dataset, batch_size=batch_size)

        return dataset, TEXT, iterator
    return dataset, TEXT


def build_corpus(df, txts: list = None, corpus: list = None):
    """Generates corpus (list of str) and vocab with occurrence count (dict of
     set of unique tokens).

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

    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Vocabulary.PAD_token: "PAD",
                           Vocabulary.SOS_token: "SOS",
                           Vocabulary.EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


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

    print('Token 4 corresponds to token:', voc.to_word(4))
    print('Token "this" corresponds to index:', voc.to_index('this'))
    for word in range(voc.num_words):
        print(voc.to_word(word))

    sent_tkns = []
    sent_idxs = []
    for word in corpus[3].split(' '):
        sent_tkns.append(word)
        sent_idxs.append(voc.to_index(word))
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
