#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: read_files.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/2/2019 15:27

import os
from gensim.utils import simple_preprocess


class ReadTxtUnderDir:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def __iter__(self):
        for file_name in os.listdir(self.dir_path):
            if not file_name.endswith('seg'):
                continue

            file_path = os.path.join(self.dir_path, file_name)
            for line in open(file_path, encoding='utf-8'):
                yield simple_preprocess(line, min_len=2)
