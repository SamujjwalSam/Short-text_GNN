from torch import load, save
from os.path import join, exists
from networkx import adjacency_matrix

from Text_Encoder.finetune_static_embeddings import create_clean_corpus
from File_Handlers.csv_handler import read_csv
from File_Handlers.json_handler import save_json, read_json
from Data_Handlers.token_handler_nx import Token_Dataset_nx
from Text_Processesor.tokenizer import BERT_tokenizer
from Text_Processesor.build_corpus_vocab import get_dataset_fields
from Utils.utils import count_parameters, logit2label, freq_tokens_per_class, \
    split_target, sp_coo2torch_coo, get_token2pretrained_embs
from config import configuration as cfg, platform as plat, username as user,\
    dataset_dir, emb_dir, pretrain_dir
from Logger.logger import logger


def get_subword_vocab(data_filename, data_dir=dataset_dir, tokenizer=BERT_tokenizer(),
                      min_freq=cfg['pretrain']['min_freq'], get_clean_corpus=False, low_oov_ignored=None):
    """ Creates cleaned corpus and finds oov subword tokens for BERT.

    :param read_input:
    :param path:
    :param name:
    :param min_freq:
    :return:
    """
    fullpath = join(data_dir, data_filename)
    corpus_toks, corpus_strs = None, None
    if exists(fullpath + '_corpus_toks.json')\
            and exists(fullpath + '_vocab.json'):
        vocab = read_json(fullpath + '_vocab')
        # fields = read_json(fullpath + '_fields')
        # oov_high_freqs = read_json(fullpath + '_oov_high_freqs'))
        if get_clean_corpus:
            corpus_toks = read_json(fullpath + '_corpus_toks',
                                    convert_ordereddict=False)
            corpus_strs = read_json(fullpath + '_corpus_strs',
                                    convert_ordereddict=False)
    else:
        dataset, (fields, LABEL) = get_dataset_fields(
            csv_dir='', csv_file=join(data_dir, data_filename + ".csv"),
            min_freq=min_freq, tokenizer=tokenizer.tokenize, fix_len=None)

        fields.vocab.stoi2 = dict(fields.vocab.stoi)

        vocab = {
            'freqs':       fields.vocab.freqs,
            'str2idx_map': dict(fields.vocab.stoi),
            'idx2str_map': fields.vocab.itos,
        }
        save_json(vocab, fullpath + '_vocab')
        # save_json(fields, fullpath + '_fields')

        if get_clean_corpus:
            if low_oov_ignored is None:
                low_oov_ignored = list(set(vocab['freqs'].keys()) - set(vocab['idx2str_map']))
            corpus_strs, corpus_toks, ignored_examples =\
                create_clean_corpus(dataset, low_oov_ignored=low_oov_ignored)

            # save_json(oov_high_freqs, fullpath + '_oov_high_freqs'))
            save_json(corpus_strs, fullpath + '_corpus_strs')
            save_json(corpus_toks, fullpath + '_corpus_toks')

    return vocab, corpus_toks, corpus_strs


def create_token_graph(joint_path, all_source_path, target_path,
                       min_freq=cfg['pretrain']['min_freq'], tokenizer=BERT_tokenizer()):
    joint_vocab, joint_corpus_toks, joint_corpus_strs =\
        get_subword_vocab(joint_path, min_freq=min_freq, tokenizer=tokenizer)

    low_oov_ignored = list(set(joint_vocab['freqs'].keys()) - set(joint_vocab['idx2str_map']))

    source_vocab, source_corpus_toks, source_corpus_strs =\
        get_subword_vocab(all_source_path, min_freq=min_freq, tokenizer=tokenizer,
                          get_clean_corpus=True, low_oov_ignored=low_oov_ignored)

    target_vocab, target_corpus_toks, target_corpus_strs =\
        get_subword_vocab(target_path, min_freq=min_freq, tokenizer=tokenizer,
                          get_clean_corpus=True, low_oov_ignored=low_oov_ignored)

    global_token_graph = Token_Dataset_nx(
        [source_corpus_toks, target_corpus_toks], joint_vocab,
        dataset_name=joint_path, S_vocab=source_vocab, T_vocab=target_vocab,
        window_size=3)

    # global_token_graph.add_edge_weights_glen()
    G_node_list = list(global_token_graph.G.nodes)
    logger.info(f"Number of nodes {len(G_node_list)} and edges "
                f"{len(global_token_graph.G.edges)} in global token graph")

    # global_token_graph.save_graph()

    return global_token_graph, joint_vocab, joint_corpus_toks, source_vocab, target_vocab


def get_glen_token_graph_data(labelled_source_name, unlabelled_source_name,
                              unlabelled_target_name, source_lablled_df=None,
                              data_dir=dataset_dir, tokenizer=BERT_tokenizer()):
    if source_lablled_df is None:
        source_lablled_df = read_csv(data_dir, labelled_source_name)
    s_unlab_df = read_csv(data_file=unlabelled_source_name, data_dir=data_dir)
    all_source_df = s_unlab_df.append(source_lablled_df[['text']])
    all_source_dataname = 'all_source_' + labelled_source_name
    all_source_df.to_csv(join(data_dir, all_source_dataname + ".csv"))

    t_unlab_df = read_csv(data_dir, unlabelled_target_name)
    # t_unlab_df.rename(columns={'tweets': 'text'}, inplace=True)
    joint_df = s_unlab_df.append(t_unlab_df)
    joint_dataname = 'joint_' + cfg['data']['name']
    joint_df.to_csv(join(data_dir, joint_dataname + ".csv"))

    global_token_graph, joint_vocab, joint_corpus_toks, source_vocab, target_vocab =\
        create_token_graph(joint_path=joint_dataname, all_source_path=all_source_dataname,
                           target_path=unlabelled_target_name, tokenizer=tokenizer)

    return global_token_graph, joint_vocab, joint_corpus_toks, source_vocab, target_vocab


def get_node_vectors(G, s2i, X):
    node_txt2embs = {}
    for node_id, data in G.nodes(data=True):
        if data['node_txt'] == '<unk>' or data['node_txt'] == '<pad>':
            continue
        node_txt2embs[data['node_txt']] = X[s2i[data['node_txt']]]
    return node_txt2embs


def construct_token_graph(init_embs, vocab, labelled_source_name, unlabelled_source_name,
                          unlabelled_target_name, train_df=None, tokenizer=BERT_tokenizer(),
                          data_dir=dataset_dir, use_lpa=False):
    # if exists(labelled_source_path + 'S_vocab.json')\
    #         and exists(labelled_source_path + 'T_vocab.json')\
    #         and exists(labelled_source_path + 'labelled_token2vec_map.json'):
    #     C_vocab = read_json(labelled_source_path + 'C_vocab')
    #     S_vocab = read_json(labelled_source_path + 'S_vocab')
    #     T_vocab = read_json(labelled_source_path + 'T_vocab')
    #     labelled_token2vec_map = read_json(labelled_source_path + 'labelled_token2vec_map')
    #
    #     if not exists(labelled_source_path + '_high_oov_freqs.json'):
    #         S_dataset, (S_fields, LABEL) = get_dataset_fields(
    #             csv_dir=data_dir, csv_file=unlabelled_source_name + ".csv")
    #         T_dataset, (T_fields, LABEL) = get_dataset_fields(
    #             csv_dir=data_dir, csv_file=unlabelled_target_name + ".csv")
    # else:
    #     C_vocab, C_dataset, S_vocab, S_dataset, S_fields, T_vocab,\
    #     T_dataset, T_fields, labelled_token2vec_map, _ =\
    #         create_glen_vocab(s_lab_df=self.lab_source_df, data_dir=data_dir,
    #                           labelled_source_name=labelled_source_name,
    #                           unlabelled_source_name=unlabelled_source_name,
    #                           unlabelled_target_name=unlabelled_target_name)
    #     ## Save vocabs:
    #     save_json(C_vocab, labelled_source_path + 'C_vocab')
    #     save_json(S_vocab, labelled_source_path + 'S_vocab')
    #     save_json(T_vocab, labelled_source_path + 'T_vocab')
    #     save_json(labelled_token2vec_map, labelled_source_path + 'labelled_token2vec_map')
    #
    # if exists(labelled_source_path + '_high_oov_freqs.json')\
    #         and exists(labelled_source_path + '_corpus.json')\
    #         and exists(labelled_source_path + '_corpus_toks.json'):
    #     high_oov_freqs = read_json(labelled_source_path + '_high_oov_freqs')
    #     # low_glove_freqs = read_json(labelled_source_name+'_low_glove_freqs')
    #     corpus = read_json(labelled_source_path + '_corpus', convert_ordereddict=False)
    #     corpus_toks = read_json(labelled_source_path + '_corpus_toks', convert_ordereddict=False)
    # else:
    #     ## Get all OOVs which does not have Glove embedding:
    #     high_oov_freqs, low_glove_freqs, corpus, corpus_toks =\
    #         preprocess_and_find_oov(
    #             (S_dataset, T_dataset), C_vocab, glove_embs=glove_embs,
    #             labelled_vocab_set=set(labelled_token2vec_map.keys()))
    #
    #     # oov_high_freqs, glove_low_freqs, oov_low_freqs =\
    #     #     preprocess_and_find_oov2(vocab, glove_embs=glove_embs, labelled_vocab_set=set(
    #     #         vocab['str2idx_map'].keys()), add_glove_tokens_back=False)
    #     #
    #     # corpus_strs, corpus_toks, ignored_examples =\
    #     #     create_clean_corpus(dataset, low_oov_ignored=oov_low_freqs)
    #
    #     ## Save token sets: high_oov_freqs, low_glove_freqs, corpus, corpus_toks
    #     save_json(high_oov_freqs, labelled_source_path + '_high_oov_freqs')
    #     # save_json(low_glove_freqs, labelled_source_name+'_low_glove_freqs', overwrite=True)
    #     save_json(corpus, labelled_source_path + '_corpus')
    #     save_json(corpus_toks, labelled_source_path + '_corpus_toks')
    #     save_json(C_vocab, labelled_source_path + 'C_vocab', overwrite=True)

    ## Create token graph:
    logger.info(f'Creating token graph:')
    g_ob, joint_vocab, joint_corpus_toks, source_vocab, target_vocab =\
        get_glen_token_graph_data(
            labelled_source_name, unlabelled_source_name,
            unlabelled_target_name, source_lablled_df=train_df,
            tokenizer=tokenizer)

    node_list = list(g_ob.G.nodes)
    logger.info(f"Number of nodes {len(node_list)} and edges "
                f"{len(g_ob.G.edges)} in token graph")

    ## Get adjacency matrix and node embeddings in same order:
    logger.info('Accessing token adjacency matrix')
    ## Note: Saving sparse tensor usually gets corrupted.
    # adj_filename = join(data_dir, labelled_source_name + "_adj.pt")
    # if exists(adj_filename):
    #     adj = load(adj_filename)
    #     # adj = sp_coo2torch_coo(adj)
    # else:
    #     adj = adjacency_matrix(G, nodelist=node_list, weight='weight')
    #     adj = sp_coo2torch_coo(adj)
    #     save(adj, adj_filename)
    A = adjacency_matrix(g_ob.G, nodelist=node_list, weight='weight')
    A = sp_coo2torch_coo(A)

    logger.info('Accessing token graph node vectors:')
    emb_filename = join(data_dir, labelled_source_name + "_subword_emb.pt")
    if exists(emb_filename):
        X = load(emb_filename)
    else:
        # init_embs_dict = get_token2pretrained_embs(init_embs, node_list, vocab)
        init_embs_dict = get_node_vectors(g_ob.G, vocab, init_embs)
        ## Use initial BERT embeddings as token graph node representation:
        logger.info('Get node vectors from token graph:')
        X = g_ob.get_node_embeddings_from_dict(init_embs_dict, init_embs_dict,
                                               joint_vocab['idx2str_map'])
        # X = sp_coo2torch_coo(X)
        save(X, emb_filename)

    ## Normalize Adjacency matrix:
    logger.info('Normalize token graph:')
    A = g_ob.normalize_adj(A)

    return g_ob, A, X, joint_vocab
