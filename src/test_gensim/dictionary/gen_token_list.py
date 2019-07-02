#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: gen_token_list.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/2/2019 17:16
from gensim.utils import simple_preprocess


def tokens_file2token_list(tokens_file, min_len=1):
    with open(tokens_file, 'r', encoding='utf-8') as f:
        return [simple_preprocess(line, min_len=min_len) for line in f]


def docs_list2token_list(documents, min_len=1):
    # return [[text for text in doc.split()] for doc in documents]
    return [simple_preprocess(doc, min_len=min_len) for doc in documents]
