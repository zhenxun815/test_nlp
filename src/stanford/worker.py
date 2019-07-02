#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: worker.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 6/27/2019 16:26

import re
import string

from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from os import path


class SegmentWorker:

    def __init__(self):
        file_path = path.realpath(__file__)
        dir_path = path.dirname(file_path)
        self.path_to_jar = path.join(dir_path, 'stanford-segmenter-3.9.2.jar')
        self.path_to_model = path.join(dir_path, 'data/ctb.gz')  # pku.gz
        self.path_to_dict = path.join(dir_path, 'data/dict-chris6.ser.gz')
        self.path_to_sihan_corpora_dict = path.join(dir_path, 'data/')
        self.seg = StanfordSegmenter(
            path_to_jar=self.path_to_jar,
            java_class='edu.stanford.nlp.ie.crf.CRFClassifier',
            path_to_model=self.path_to_model,
            path_to_dict=self.path_to_dict,
            path_to_sihan_corpora_dict=self.path_to_sihan_corpora_dict)

    def seg_file(self, file_to_segment):
        """segment a file and return the result string"""
        seg_result = self.seg.segment_file(file_to_segment)

        translator = str.maketrans('', '', string.digits)
        seg_result = seg_result.translate(translator)
        seg_result = re.sub('[\\\\.!/_,$%^*(+\\"\']+|[+—！，：；。？、~@#￥%…&*（）]+', '', seg_result)
        # print(seg_result)
        return seg_result

    def seg_file2list(self, file_to_segment):
        """segment a text file and return array of tokens"""
        seg_result = self.seg_file(file_to_segment)
        # print(seg_result)
        return seg_result.split()

    def seg_file2file(self, origin_file, dest_file):
        """segment a text file and write result tokens to another file"""
        seg_result = self.seg_file(origin_file)
        seg_result = re.sub('\\s+', ' ', seg_result)
        # print(seg_result)
        with open(dest_file, 'w', encoding='UTF-8') as f:
            f.write(seg_result)
