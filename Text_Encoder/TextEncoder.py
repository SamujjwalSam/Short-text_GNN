# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Class to process and load pretrained models.

__description__ : Class to process and load pretrained models.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license
found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : TextEncoder

__variables__   :

__methods__     :
"""

import numpy as np
from os import mkdir
from os.path import join, exists, split
from collections import OrderedDict

# import gensim
from gensim.models import word2vec, doc2vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
# from gensim.models.keyedvectors import Doc2VecKeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess

from config import configuration as cfg, pretrain_dir, emb_dir
from config import platform as plat
from config import username as user
from Logger.logger import logger


# from Text_Processesor.clean_text import Clean_Text


class TextEncoder:
    """
    Class to process and load pretrained models.

    Supported models: glove, word2vec, fasttext, googlenews, bert, lex, etc.
    """

    def __init__(self, model_type: str = "googlenews",
                 model_dir: str = pretrain_dir,
                 embedding_dim: int = cfg['embeddings']['emb_dim']):
        """
        Initializes the pretrain class and checks for paths validity.

        Args:
            model_type : Path to the file containing the html files.
            Supported model_types:
                glove (default)
                word2vec
                fasttext_wiki
                fasttext_crawl
                fasttext_wiki_subword
                fasttext_crawl_subword
                lex_crawl
                lex_crawl_subword
                googlenews
                bert_multi
                bert_large_uncased
        """
        super(TextEncoder, self).__init__()
        self.model_type = model_type
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        # self.clean = Clean_Text()

        if model_type == "googlenews":
            filename = "GoogleNews-vectors-negative300.bin"
            binary_file = True
        elif model_type == "glove":
            filename = "glove.6B.300d.txt"
            binary_file = False
        elif model_type == "fasttext_wiki":
            filename = "wiki-news-300d-1M.vec"
            binary_file = False
        elif model_type == "fasttext_crawl":
            filename = "crawl-300d-2M.vec.zip"
            binary_file = False
        elif model_type == "fasttext_wiki_subword":
            filename = "wiki-news-300d-1M-subword.vec.zip"
            binary_file = False
        elif model_type == "fasttext_crawl_subword":
            filename = "crawl-300d-2M-subword.vec.zip"
            binary_file = False
        elif model_type == "lex_crawl":
            filename = "lexvec.commoncrawl.300d.W+C.pos.vectors.gz"
            binary_file = True
        elif model_type == "lex_crawl_subword":
            filename = "lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin.gz"
            binary_file = True
        elif model_type == "bert_multi":
            filename = "BERT_multilingual_L-12_H-768_A-12.zip"
            binary_file = True
        elif model_type == "bert_large_uncased":
            filename = "BERT_large_uncased_L-24_H-1024_A-16.zip"
            binary_file = True
        else:
            raise Exception(
                "Unknown pretrained model type: [{}]".format(model_type))
        # logger.debug("Creating TextEncoder.")
        self.model_file_name = filename
        self.binary = binary_file
        self.pretrain_model = None
        # self.pretrain_model = self.load_word2vec(self.model_dir,
        # model_file_name=self.model_file_name, model_type=model_type)

    def load_doc2vec(
            self, documents: list, vector_size: int = cfg['embeddings']['emb_dim'],
            window: int = cfg["prep_vecs"]["window"], min_count: int = cfg["prep_vecs"]["min_freq"],
            workers: int = cfg["text_process"]["workers"], seed: int = 0,
            clean_tmp: bool = False, save_model: bool = True,
            doc2vec_model_file: str = cfg["data"]["name"] + "_doc2vec",
            doc2vec_dir: str = join(cfg["paths"]['dataset_root'][plat][user], cfg["data"]["name"]),
            negative: int = cfg["prep_vecs"]["negative"]) -> doc2vec.Doc2Vec:
        """
        Generates vectors from documents.

        https://radimrehurek.com/gensim/models/doc2vec.html

        :param save_model:
        :param clean_tmp: Flag to set if cleaning is to be done.
        :param doc2vec_dir:
        :param doc2vec_model_file: Name of Doc2Vec model.
        :param negative: If > 0, negative sampling will be used, the int for
        negative specifies how many “noise words” should be drawn (usually
        between 5-20).
        :param documents:
        :param vector_size:
        :param window:
        :param min_count:
        :param workers:
        :param seed:
        """
        full_model_name = doc2vec_model_file + "_" + str(
            vector_size) + "_" + str(window) + "_" + str(min_count) + "_"\
                          + str(negative)
        if exists(join(doc2vec_dir, full_model_name)):
            logger.info(
                "Loading doc2vec model [{}] from: [{}]".format(full_model_name,
                                                               doc2vec_dir))
            doc2vec_model = doc2vec.Doc2Vec.load(
                join(doc2vec_dir, full_model_name))
        else:
            train_corpus = list(self.read_corpus(documents))
            doc2vec_model = doc2vec.Doc2Vec(train_corpus,
                                            vector_size=vector_size,
                                            window=window, min_count=min_count,
                                            workers=workers, seed=seed,
                                            negative=negative)
            # doc2vec_model.build_vocab(train_corpus)
            doc2vec_model.train(train_corpus,
                                total_examples=doc2vec_model.corpus_count,
                                epoch=doc2vec_model.epoch)
            if save_model:
                save_path = get_tmpfile(join(doc2vec_dir, full_model_name))
                doc2vec_model.save(save_path)
                logger.info("Saved doc2vec model to: [{}]".format(save_path))
            if clean_tmp:  # Do this when finished training a model (no more
                # updates, only querying, reduce memory usage)
                doc2vec_model.delete_temporary_training_data(
                    keep_doctags_vectors=True, keep_inference=True)
        return doc2vec_model

    def read_corpus(self, documents, tokens_only=False):
        """
        Read the documents, pre-process each line using a simple gensim
        pre-processing tool and return a list of words. The tag is simply the
        zero-based line number.

        :param documents: List of documents.
        :param tokens_only:
        """
        for i, line in enumerate(documents):
            if tokens_only:
                yield simple_preprocess(line)
            else:  # For training data, add tags, tags are simply zero-based
                # line number.
                yield doc2vec.TaggedDocument(simple_preprocess(line), [i])

    def get_doc2vecs(self, documents: list, doc2vec_model=None):
        """
        Generates vectors for documents.

        :param doc2vec_model: doc2vec model object.
        :param documents:
        :return:
        """
        if doc2vec_model is None:  # If model is not supplied, create model.
            doc2vec_model = self.load_doc2vec(documents)
        doc2vectors = []
        for doc in documents:
            # Infer vector for a new document:
            doc2vectors.append(doc2vec_model.infer_vector(self.clean.tokenizer_spacy(doc)))
        # Converting Dict values to Numpy array:
        doc2vectors = np.asarray(list(doc2vectors))
        return doc2vectors

    def load_word2vec(
            self, model_dir: str = pretrain_dir,
            model_type: str = 'googlenews', encoding: str = 'utf-8',
            model_file_name: str = "GoogleNews-vectors-negative300.bin",
            newline: str = '\n', errors: str = 'ignore'):
        """
        Loads Word2Vec model and returns initial weights for embedding layer.

        inputs:
        model_type      # GoogleNews / glove
        in_dim    # Word vector dimensionality
        """
        if self.pretrain_model is None:
            logger.debug(f"Using [{model_type}] model from [{join(model_dir, model_file_name)}]")
            if model_type == 'googlenews' or model_type == "fasttext_wiki":
                assert (exists(join(model_dir, model_file_name)))
                if exists(join(model_dir, model_file_name + '.bin')):
                    try:
                        pretrain_model = FastText.load_fasttext_format(
                            join(model_dir, model_file_name + '.bin'))  # For original
                        # fasttext *.bin format.
                    except Exception as e:
                        pretrain_model = KeyedVectors.load_word2vec_format(
                            join(model_dir, model_file_name + '.bin'), binary=True)
                else:
                    try:
                        pretrain_model = KeyedVectors.load_word2vec_format(
                            join(model_dir, model_file_name), binary=self.binary)
                    except Exception as e:  # On exception, trying a
                        # different format.
                        logger.info(
                            'Loading original word2vec format failed. Trying '
                            'Gensim format.')
                        pretrain_model = KeyedVectors.load(
                            join(model_dir, model_file_name))
                    pretrain_model.save_word2vec_format(
                        join(model_dir, model_file_name + ".bin"),
                        binary=True)  # Save model in binary format for faster loading in future.
                    logger.info("Saved binary model at: [{0}]".format(
                        join(model_dir, model_file_name + ".bin")))
                    logger.info(type(pretrain_model))
            elif model_type == 'glove':
                assert (exists(join(model_dir, model_file_name)))
                logger.info('Loading existing Glove model: [{0}]'.format(
                    join(model_dir, model_file_name)))
                ## dictionary, where key is word, value is word vectors
                pretrain_model = OrderedDict()
                for line in open(join(model_dir, model_file_name),
                                 encoding=encoding):
                    tmp = line.strip().split()
                    word, vec = tmp[0], map(float, tmp[1:])
                    assert (len(vec) == self.embedding_dim)
                    if word not in pretrain_model:
                        pretrain_model[word] = vec
                logger.info(f'Found [{len(pretrain_model)}] word vectors.')
                assert (len(pretrain_model) == 400000)
            else:
                raise ValueError(f'Unknown pretrain model type: {model_type}')
            self.pretrain_model = pretrain_model
            return self.pretrain_model
        else:
            return self.pretrain_model

    def get_embedding_matrix(self, vocabulary_inv: dict):
        """
        Generates the embedding matrix.

        :param vocabulary_inv:
        :param embedding_model:
        :return:
        """
        embedding_weights = [self.pretrain_model[w] if w in self.pretrain_model
                             else np.random.uniform(-0.25, 0.25, self.embedding_dim)
                             for w in vocabulary_inv]
        embedding_weights = np.array(embedding_weights).astype('float32')

        return embedding_weights


def load_word2vec(model_dir: str = emb_dir,
                  model_file_name: str = 'crisisNLP_word_vector', binary=True):
    """ Loads Word2Vec model. """
    model_path = join(model_dir, model_file_name)
    assert exists(model_path), f'Model [{model_path}] not found'
    try:
        pretrain_model = FastText.load_fasttext_format(model_path)
    except Exception as e:
        logger.info('Loading fasttext format failed; trying word2vec format.')
        try:
            pretrain_model = KeyedVectors.load_word2vec_format(
                model_path, binary=binary)
        except Exception as e:
            logger.info('Loading word2vec format failed; trying Gensim format.')
            pretrain_model = KeyedVectors.load(model_path)
            # pretrain_model.save_word2vec_format(
            #     join(model_dir, model_file_name + "_w2v.bin"),
            #     binary=True)  # Save model in binary format for faster loading in future.
            # logger.info("Saved binary model at: [{0}]".format(
            #     join(model_dir, model_file_name + ".bin")))
    logger.debug(f"Loaded model from [{model_path}]")

    return pretrain_model


def train_w2v(sentences, tokens_list, in_dim=cfg['embeddings']['emb_dim'],
              min_freq=cfg["prep_vecs"]["min_freq"], context=cfg["prep_vecs"]["window"]):
    """ Trains, saves, loads Word2Vec model.

    :param sentences: list of tokenized tokens
    :param in_dim: Word vector dimension
    :param min_freq: Minimum word count
    :param context: Context window size
    :return: Returns initial weights for embedding layer.
    """
    model_name = "w2v_{:d}features_{:d}minfreq_{:d}context".format(
        in_dim, min_freq, context)
    model_dir = join(pretrain_dir, 'w2v')
    model_path = join(model_dir, model_name)
    if exists(model_path):
        pretrain_model = word2vec.Word2Vec.load(model_path)
        logger.debug(f'Loaded existing Word2Vec model from: [{model_path}]')
    else:
        ## Set values for various parameters
        num_workers = 8  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        ## Initialize and train the model
        logger.info("Training Word2Vec model...")
        # sentences = [[vocabulary_inv[w] for w in s] for s in
        #              sentence_matrix]
        pretrain_model = word2vec.Word2Vec(
            sentences, workers=num_workers, size=in_dim,
            min_count=min_freq, window=context, sample=downsampling)

        ## If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        pretrain_model.init_sims(replace=True)

        if not exists(model_dir):
            mkdir(model_dir)
        pretrain_model.save(model_path)
        logger.info(f'Saved Word2Vec model at [{model_path}]')

    ## add unknown words
    embedding_weights = [np.array(
        [pretrain_model[w] if w in pretrain_model else np.random.uniform(
            -0.25, 0.25, pretrain_model.vector_size) for w in tokens_list])][0]
    return embedding_weights


if __name__ == '__main__':
    sentences = ['Obama speaks to the media in Illinois',
                 'The president greets the press in Chicago']
    tokens_set = set()
    sentences_toks = []
    for sent in sentences:
        tokens = sent.lower().split()
        sentences_toks.append(tokens)
        tokens_set.update(tokens)
    tokens_list = list(tokens_set)
    vectors = train_w2v(sentences_toks, tokens_list)
    logger.debug(vectors)
