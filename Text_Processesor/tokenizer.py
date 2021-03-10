# coding=utf-8
# !/usr/bin/python3.7  # Please use python 3.7
"""
__synopsis__    : Tokenize raw text
__description__ : Train or use a Tokenizer from huggingface tokenizers.
__project__     : GCPD
__classes__     : GCPD
__variables__   :
__methods__     :
__author__      : Samujjwal
__version__     : ":  "
__date__        : "04/03/21"
__last_modified__:
__copyright__   : "Copyright (c) 2020, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license
                   found in the LICENSE file in the root directory of this
                   source tree."
"""

import requests
from typing import List, Dict
from os.path import exists, join
from tokenizers import BertWordPieceTokenizer
# from transformers import BertTokenizer


class BERT_tokenizer():
    """ BERT tokenizer class """
    def __init__(self, vocab_name: str = 'bert-base-uncased-vocab.txt',
                 lowercase: bool = True):
        if not exists(vocab_name):
            ## Download vocab from:
            url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
            bert_vocab = requests.get(url)
            open(vocab_name, 'wb').write(bert_vocab.content)

        self.tokenizer = BertWordPieceTokenizer(vocab_name, lowercase=lowercase)

    def tokenize_batch(self, txts: List[str]):
        # txts = ["Hello, y'all!", "Hello to you too!"]
        encs = self.tokenizer.encode_batch(txts)
        tokenized_txts = []
        for enc in encs:
            tokenized_txts.append(enc.tokens)
        return tokenized_txts

    def tokenize(self, txt: str):
        # txt = "Hello, y'all!"
        return self.tokenizer.encode(txt).tokens


def tokenize_txts(txts):
    tk = BERT_tokenizer()
    enc = tk.tokenize(txts[0])
    print(enc)
    encs = tk.tokenize_batch(txts)
    print(encs)


if __name__ == "__main__":
    txts = ['I am Samujjwal', 'I am happy']
    tokenize_txts(txts)
