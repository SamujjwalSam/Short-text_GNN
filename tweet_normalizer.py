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
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from emoji import demojize

from File_Handlers.json_handler import read_json

acronym = read_json()


def stemming(word, lemmatize=False):
    if lemmatize:
        wnl = WordNetLemmatizer()
        return wnl.lemmatize(word)
    else:
        ps = PorterStemmer()
        return ps.stem(word)


def digit_count_str(s):
    return len(str(abs(int(s))))


def is_float(w):
    """takes str and returns if it is a decimal number"""
    try:
        num = float(w)
        return True, num
    except ValueError as e:
        return False, w


def find_numbers(text, replace=True,
                 numexp = re.compile(r'(?:(?:\d+,?)+(?:\.?\d+)?)')):
    """

    :param text: strings that contains digit and words
    :param replace: bool to decide if numbers need to be replaced.
    :return: text, list of numbers
    Ex:
    '1230,1485': 8d
    '-2': 1d
    3.0 : 2f
    """
    # numbers = numexp.findall(" ".join(text))
    numbers = numexp.findall(text)
    # logger.debug(numbers)
    if replace:
        for num in numbers:
            try:
                i = text.index(num)
                if num.isdigit():
                    text[i] = str(len(num)) + "d"
                else:
                    try:
                        num = float(num)
                        text[i] = str(len(str(num)) - 1) + "f"
                    except ValueError as e:
                        if len(num) > 9:
                            text[i] = "phonenumber"
                        else:
                            text[i] = str(len(num) - 1) + "d"
            except ValueError as e:
                pass
                # logger.debug(("Could not find number [",num,"] in tweet: [
                # ",text,"]")
    return text, numbers


def normalizeToken(token):
    token_lower = token
    # token_lower2, _ = find_numbers(token_lower)

    ## Replace token with acronym:
    try:
        token = acronym[token]
    except KeyError:
        pass

    if token.startswith("@"):
        return "@USER"
    elif token.startswith("#"):
        return "#HASH "+token[1:]
    elif token_lower.startswith("http") or token_lower.startswith("www"):
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


def normalizeTweet(tweet, tokenizer=TweetTokenizer(), return_tokens=False,
                   lower_case=True):
    # tweet2, _ = find_numbers(tweet)
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    if lower_case:
        normTweet = normTweet.lower()

    normTweet = normTweet.replace("cannot ", "can not ")\
        .replace("n't ", " not ")\
        .replace("n 't ", " not ")\
        .replace("ca n't", "can not")\
        .replace("ai n't", "is not")
    normTweet = normTweet.replace("'m ", " 'm ")\
        .replace("'re ", " are ")\
        .replace("'s ", " 's ")\
        .replace("'ll ", " will ")\
        .replace("'d ", " 'd ")\
        .replace("'ve ", " have ")
    normTweet = normTweet.replace(" p . m .", "  p.m.")\
        .replace(" p . m ", " pm ")\
        .replace(" a . m .", " am.")\
        .replace(" a . m ", " am ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    if return_tokens:
        return normTweet.split()

    return " ".join(normTweet.split())


if __name__ == "__main__":
    t1 = "SC has first two #presumptive cases of coronavirus, DHEC confirms "\
         "https://postandcourier.com/health/covid19/sc-has-first-two"\
         "-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae"\
         "-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source="\
         "twitter&utm_campaign=user-share… via @postandcourier"

    t2 = "$ #India dispatched 100,000 bottles of #RailNeer water 959-5085116"\
         " to quake-hit #Nepal on Saturday night. http://t.co/HXkVtw9hRo "\
         "#nepal via @oneindia"

    # print(normalizeTweet(t1))
    print(normalizeTweet(t2))
    print(normalizeTweet(t2, return_tokens=True))
