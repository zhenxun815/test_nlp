#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: corpus_utils.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/4/2019 15:56
import os
from gensim.corpora import Dictionary, MmCorpus
from test_gensim.common_utils.token_utils import get_tokens_of_file
from test_gensim.common_utils.token_utils import get_tokens_of_doc


def get_bow_of_file(file_path, dct: Dictionary, human_read: bool):
    """
    Note that the file content must be segmented.
    If human_read is True, each element of the list of corpus will be human readable,
    like ('白菜',1), that means the id of token was replaced by the token word.
    :param file_path:
    :param dct:
    :param human_read:
    :return:
    """
    file_tokens = get_tokens_of_file(file_path)
    bow = dct.doc2bow(file_tokens)
    return trans_bow_human_readable(bow, dct) if human_read else bow


def get_bow_of_doc(doc: str, dct: Dictionary, human_read: bool):
    """
    Note that the doc content must be segmented.
    If human_read is True, each element of the list of corpus will be human readable,
    like ('白菜',1), that means the id of token was replaced by the token word.
    :param doc:
    :param dct:
    :param human_read:
    :return:
    """
    doc_tokens = get_tokens_of_doc(doc)
    bow = dct.doc2bow(doc_tokens)
    return trans_bow_human_readable(bow, dct) if human_read else bow


def save2disk(save_path, corpus):
    if not os.path.exists(save_path):
        open(save_path, 'w', encoding='utf-8').close()

    MmCorpus.serialize(save_path, corpus)


def load_from_disk(load_path):
    return MmCorpus(load_path)


def trans_bow_human_readable(bow, dct: Dictionary):
    return [(dct[index], count) for index, count in bow]
