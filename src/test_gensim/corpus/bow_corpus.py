#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: generate dictionary from files under given dir and get corpus(bag of words)
# of each file using the prior mentioned dictionary
# @File: bow_corpus.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/4/2019 11:03

import os

from test_gensim.common_utils.dct_utils import gen_dictionary_of_dir
from test_gensim.common_utils.token_utils import get_tokens_of_file


class BoWCorpus:
    """Generator class, each time yield a 2 elements tuple, 1st element is file name,
    the 2nd element is corpus of the file.
    """

    def __init__(self, dictionary, dir_path):
        self.dictionary = dictionary
        self.dir_path = dir_path

    def __iter__(self):
        for file_name in os.listdir(self.dir_path):
            if not file_name.endswith('seg'):
                continue

            file_path = os.path.join(self.dir_path, file_name)
            file_tokens = get_tokens_of_file(file_path)
            yield (file_name, self.dictionary.doc2bow(file_tokens))


if __name__ == '__main__':
    dir_dictionary = gen_dictionary_of_dir('../resources')
    for file_corpus in BoWCorpus(dir_dictionary, '../resources'):
        file_name = file_corpus[0]
        corpus = [(dir_dictionary[index], count) for index, count in file_corpus[1]]
        print('file_name is: {}'.format(file_name))
        print('corpus is {}'.format(corpus))
