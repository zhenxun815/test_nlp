#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: read_files.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/2/2019 15:27

import os
from gensim.utils import simple_preprocess


class ReadTxt:
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


def preprocess_dir(dir_path):
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
        file_tokens_list = preprocess_file(file_path)
        file_tokens[file_path] = file_tokens_list
    return file_tokens


def preprocess_file(file_path):
    """
    Get list of tokens of the given file
    :param file_path: file to be processed
    :return: list of tokens
    """
    line_tokens_lists = ReadTxt(file_path)
    # flatten the 2D tokens list
    return [token for tokens in line_tokens_lists for token in tokens]
