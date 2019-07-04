#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: token_utils.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/4/2019 15:34
import os
from gensim.utils import simple_preprocess


class PreprocessFile:
    """
        Read file line by line and parse each line to a tokens list.
        Note that the generated list is a 2D list, each element list in it
        consist of the tokens of each line.
        """

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        for line in open(self.file_path, encoding='utf-8'):
            yield simple_preprocess(line, min_len=2)


def get_tokens_of_file(file_path) -> list:
    """
    Get list of tokens of the given file
    :param file_path: file to be processed
    :return: list of tokens
    """
    line_tokens_lists = PreprocessFile(file_path)
    # flatten the 2D tokens list
    return [token for tokens in line_tokens_lists for token in tokens]


def get_file_tokens_of_dir(dir_path) -> list:
    """
    Get tokens of all file under the given dir path
    :param dir_path:
    :return: dict, item key is file path, value is list of tokens
    """
    file_tokens = dict()
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('seg'):
            continue
        file_path = os.path.join(dir_path, file_name)
        file_tokens_list = get_tokens_of_file(file_path)
        file_tokens[file_path] = file_tokens_list
    return file_tokens


def get_token_lists_of_docs(docs: list, min_len=1) -> list:
    """
    Trans doc list to 2D list consist of each doc's token list
    :param docs:
    :param min_len:
    :return:
    """
    return [get_tokens_of_doc(doc, min_len) for doc in docs]


def get_tokens_of_docs(docs: list, min_len=1) -> list:
    """
    Trans doc list to 1D token list
    :param docs:
    :param min_len:
    :return:
    """
    # return [[text for text in doc.split()] for doc in documents]
    tokens_list = get_token_lists_of_docs(docs, min_len)
    return [token for tokens in tokens_list for token in tokens]


def get_tokens_of_doc(doc: str, min_len=1) -> list:
    return simple_preprocess(doc, min_len=min_len)
