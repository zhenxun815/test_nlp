#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_stanford_segmenter.py.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/1/2019 14:48

from stanford.worker import StanfordWorker


def test_stanford_segmenter():
    worker = StanfordWorker()
    tokens = worker.segment_file('../resources/CN105253527A')
    for token in tokens:
        print(token)


test_stanford_segmenter()
