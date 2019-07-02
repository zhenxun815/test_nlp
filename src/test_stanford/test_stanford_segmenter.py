#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_stanford_segmenter.py.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/1/2019 14:48

from stanford.worker import SegmentWorker


def test_seg2list():
    worker = SegmentWorker()
    tokens = worker.seg_file2list('../resources/CN104188073B')
    for i, token in enumerate(tokens):
        print('token {} is {}'.format(i, token))


def test_seg2file(origin_file, dest_file):
    worker = SegmentWorker()
    worker.seg_file2file(origin_file, dest_file)


# test_seg2list()
test_seg2file('../resources/CN105253527A', '../resources/CN105253527A.seg')
