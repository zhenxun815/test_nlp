#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_tf_idf.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/5/2019 11:50
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import numpy as np

if __name__ == '__main__':
    docs = ['一种 大头菜 自然风 脱水 设备 其 特征 在于 所述 的 大头菜 自然风 脱水 设备 主要 包括 大头菜',
            '风 脱水 架 和 大头菜 风 脱水 网袋 所述 的 大头菜 风 脱水 架 主要 包括 底座 支柱 横架 横向',
            '连接 承重杆 所述 的 底座 通过 中间 的 多边形 孔 与 支柱 的 下端 的 多边形 柱 配合 而 固定']

    tokenized_docs = [simple_preprocess(doc, min_len=2) for doc in docs]

    my_dct = Dictionary(tokenized_docs)
    print('dictionary is {}'.format(my_dct.token2id))

    corpus = [my_dct.doc2bow(doc) for doc in tokenized_docs]
    for index, bow in enumerate(corpus):
        bow = [[my_dct[index], count] for index, count in bow]
        print('bow of doc {} is'.format(index))
        print(bow)

    tf_idf_model = TfidfModel(corpus, id2word=my_dct, dictionary=my_dct, smartirs='ntc')
    for index, doc in enumerate(tf_idf_model[corpus]):
        tf_idf_info = [[my_dct[id], np.around(freq, decimals=2)] for id, freq in doc]
        print('tf-idf model info of doc {} is:'.format(index))
        print(tf_idf_info)