#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_corpus.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/2/2019 16:51
from test_gensim.dictionary.gen_token_list import *
from gensim.corpora import Dictionary


def create_corpus(docs, my_dict):
    token_list = docs_list2token_list(docs)

    print('before call doc2bow, dictionary is {}'.format(my_dict.token2id))
    corpus = [my_dict.doc2bow(doc, allow_update=True) for doc in token_list]
    print('after call doc2bow, dictionary is {}'.format(my_dict.token2id))
    print('curpos is {}'.format(corpus))


dictionary = Dictionary()
documents1 = ['一种 大头菜 自然风', '风 主要 包括 大头菜 风', '架 主要 包括 底座 支柱']
create_corpus(documents1, dictionary)
documents2 = ['一种 海绵拔 轮', '支架 上 设有 海绵 辊轴', '同步 皮带轮 二 海绵 辊轴']
create_corpus(documents2, dictionary)
