#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_bigrams_trigrams.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/8/2019 14:55
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases

if __name__ == '__main__':
    docs = ['一种 大 头菜 自然风 脱水 设备 其 特征 在于 所述 的 大 头菜 自然风 脱水 设备 主要 包括 大 头菜',
            '风 脱水 架 和 大 头菜 风 脱水 网袋 所述 的 大 头菜 风 脱水 架 主要 包括 底座 支柱 横架 横向',
            '连接 承重杆 所述 的 底座 通过 中间 的 多边形 孔 与 支柱 的 下端 的 多边形 柱 配合 而 固定']

    tokenized_docs = [simple_preprocess(doc, min_len=1) for doc in docs]
    my_dct = Dictionary(tokenized_docs)
    corpus = [my_dct.doc2bow(doc) for doc in tokenized_docs]
    bigram = Phrases(tokenized_docs, min_count=1, threshold=1, scoring='default')
    trigram = Phrases(bigram[tokenized_docs])
    print(bigram[tokenized_docs[0]])
    print(trigram[bigram[tokenized_docs[0]]])
