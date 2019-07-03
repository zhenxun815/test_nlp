#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_dictionary.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/1/2019 15:36

from gensim import corpora
from test_gensim.dictionary.gen_token_list import *
from test_gensim.dictionary.read_files import preprocess_dir


def test_gen_dict_from_list():
    documents1 = ['一种 大头菜 自然风', '主要 包括 大头菜 风', '架 主要 包括 底座 支柱']
    documents2 = ['一种 海绵拔 轮', '支架 上 设有 海绵 辊轴', '同步 皮带轮 二 海绵 辊轴']

    texts1 = docs_list2token_list(documents1)
    texts2 = docs_list2token_list(documents2)

    dict_from_list = corpora.Dictionary(texts1)
    print('after add texts1, dictionary is {}'.format(dict_from_list.token2id))

    dict_from_list.add_documents(texts2)
    print('after add texts2, dictionary is {}'.format(dict_from_list.token2id))


def test_gen_dict_from_file(seg_file):
    texts_list = tokens_file2token_list(seg_file)
    print('type is {}'.format(type(texts_list)))
    dict_from_file = corpora.Dictionary(texts_list)

    print('dict is {}'.format(dict_from_file.token2id))


# test_gen_dict_from_file('../resources/CN105253527A.seg')
# test_gen_dit_from_files('../resources/')

dir_tokens = preprocess_dir('../resources/')
my_dict = corpora.Dictionary()
for file_path, file_tokens in dir_tokens.items():
    print('file is {}, tokens is {}'.format(file_path,file_tokens))

    corpus = my_dict.doc2bow(file_tokens, allow_update=True)
    word_counts = [(my_dict[token_id], count) for token_id, count in corpus]
    print('corpus is {}'.format(word_counts))
