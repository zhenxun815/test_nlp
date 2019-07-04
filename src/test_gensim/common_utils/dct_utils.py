#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: dictionary utils
# @File: dct_utils.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/4/2019 15:23

import os
from gensim.corpora import Dictionary
from test_gensim.common_utils.token_utils import get_tokens_of_file


def gen_dictionary_of_dir(dir_path):
    """
    Generate a Dictionary obj of text files under the given dir
    :rtype: Dictionary
    :param dir_path:
    :return:
    """
    dictionary = Dictionary()
    file_tokens_list = []
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('seg'):
            continue

        file_path = os.path.join(dir_path, file_name)
        file_tokens = get_tokens_of_file(file_path)
        file_tokens_list.append(file_tokens)

    dictionary.add_documents(file_tokens_list)
    return dictionary


def save_dct2disk(save_path, dct: Dictionary):
    """
    Save Dictionary obj to disk.
    :param save_path:
    :param dct:
    :return:
    """
    dct.save(save_path)


def load_dct(dct_path) -> Dictionary:
    Dictionary.load(dct_path)
