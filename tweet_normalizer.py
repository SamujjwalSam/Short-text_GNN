# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Tokenize tweets.
__description__ : Utilizes TweetTokenizer from NLTK to process and tokenize
tweets.
__project__     : Tweet_GNN_inductive
__classes__     :
__variables__   :
__methods__     :
__author__      :
__version__     : ":  "
__date__        : "07/05/20"
__last_modified__:

__taken_from__  : https://github.com/VinAIResearch/BERTweet
"""

import re
from nltk.tokenize import TweetTokenizer
from emoji import demojize


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith(
            "www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet, tokenizer=TweetTokenizer()):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ")\
        .replace("n't ", " n't ")\
        .replace("n 't ", " n't ")\
        .replace("ca n't", "can't")\
        .replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ")\
        .replace("'re ", " 're ")\
        .replace("'s ", " 's ")\
        .replace("'ll ", " 'll ")\
        .replace("'d ", " 'd ")\
        .replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.")\
        .replace(" p . m ", " p.m ")\
        .replace(" a . m .", " a.m.")\
        .replace(" a . m ", " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())


if __name__ == "__main__":
    t1 = "SC has first two presumptive cases of coronavirus, DHEC confirms "\
         "https://postandcourier.com/health/covid19/sc-has-first-two"\
         "-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae"\
         "-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source="\
         "twitter&utm_campaign=user-share… via @postandcourier"

    t2 = "#India dispatched 100,000 bottles of #RailNeer water 959-5085116"\
         " to quake-hit #Nepal on Saturday night. http://t.co/HXkVtw9hRo "\
         "#nepal via @oneindia"

    print(normalizeTweet(t1))
    print(normalizeTweet(t2))
