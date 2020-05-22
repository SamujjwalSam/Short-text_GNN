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


def build_coupus(txts: list, corpus: list = None):
    if corpus is None:
        corpus = []
    for txt in txts:
        for token in txt:
            corpus.append(token)

    vocab_stat = Counter(corpus)

    return corpus, vocab_stat


if __name__ == "__main__":
    t1 = "SC has first two presumptive cases of coronavirus, DHEC confirms "\
         "https://postandcourier.com/health/covid19/sc-has-first-two"\
         "-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae"\
         "-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source="\
         "twitter&utm_campaign=user-share… via @postandcourier"

    t2 = "#India dispatched 100,000 bottles of #RailNeer water 959-5085116"\
         " to quake-hit #Nepal on Saturday night. http://t.co/HXkVtw9hRo "\
         "#nepal via @oneindia"

    from tweet_tokenizer import normalizeTweet

    t1_toks = normalizeTweet(t1)
    t2_toks = normalizeTweet(t2)

    txts = [t1_toks, t2_toks]

    corpus1, vocab1 = build_coupus(txts)

    print(corpus1, vocab1)

    t3 = "#India dispatched 100,000 bottles of"
    t3_toks = normalizeTweet(t3)

    corpus3, vocab3 = build_coupus([t3_toks], corpus1)

    print(corpus3, vocab3)
