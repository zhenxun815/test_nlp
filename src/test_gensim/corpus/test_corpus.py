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
from test_gensim.common_utils.corpus_utils import save2disk, load_from_disk
from test_gensim.common_utils.token_utils import get_token_lists_of_docs


def create_corpus(docs, my_dict):
    token_list = docs_list2token_list(docs)

    print('before call doc2bow, dictionary is {}'.format(my_dict.token2id))
    corpus = [my_dict.doc2bow(doc, allow_update=True) for doc in token_list]
    print('after call doc2bow, dictionary is {}'.format(my_dict.token2id))
    return corpus


def print_corpus_human_readable(dictionary, corpus):
    word_counts = [[(dictionary[token_id], count) for token_id, count in item] for item in corpus]
    print('corpus is {}'.format(word_counts))


def test_create_corpus():
    dictionary = Dictionary()
    documents1 = ['一种 大头菜 自然风', '风 主要 包括 大头菜 风', '架 主要 包括 底座 支柱']
    corpus = create_corpus(documents1, dictionary)
    print('corpus is {}'.format(corpus))
    documents2 = ['一种 海绵拔 轮', '支架 上 设有 海绵 辊轴', '同步 皮带轮 二 海绵 辊轴']
    corpus = create_corpus(documents2, dictionary)
    print('corpus is {}'.format(corpus))

    print_corpus_human_readable(dictionary, corpus)


def test_save_load():
    dct = Dictionary()
    docs = ['一种 大头菜 自然风', '风 主要 包括 大头菜 风', '架 主要 包括 底座 支柱']
    docs_token_list = get_token_lists_of_docs(docs)
    dct.add_documents(docs_token_list)
    corpus = [dct.doc2bow(['大头菜', '风', '底座'])]
    print('corpus to save is {}'.format(corpus))
    save_path = 'resources/corpus/test_corpus.mm'

    save2disk(save_path, corpus)
    load_corpus = load_from_disk(save_path)
    print('load corpus is {}'.format(load_corpus))


if __name__ == '__main__':
    test_save_load()
