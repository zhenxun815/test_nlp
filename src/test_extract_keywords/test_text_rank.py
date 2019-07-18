#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_text_rank.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/18/2019 16:49

import json
import os
import re

import jieba.analyse
import pandas as pd

"""
       TextRank权重：

            1、将待抽取关键词的文本进行分词、去停用词、筛选词性
            2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
            3、计算图中节点的PageRank，注意是无向带权图
"""


def batch(dir_path: str):
    """
    process all classified docs files under specified dir, each classified docs file will generate
    a tuple whose first element is classification str, second element is pandas data obj.
    :param dir_path:
    :return:
    """
    for file_name in os.listdir(dir_path):
        clf_str = get_clf_from_file_name(file_name)
        if clf_str:
            print('start get keywords text rank of file {}'.format(file_name))
            file_path = os.path.join(dir_path, file_name)
            docs_json = json.dumps(get_json(file_path))
            # note that the docs json pass to pandas.read_json() function must be type of json string,
            # never pass a json obj to it!
            clf_data = pd.read_json(docs_json, encoding='utf-8')
            yield (clf_str, get_keywords_text_rank(clf_data, 10))


pattern = re.compile(r'(?P<section>[A-Z])_(?P<main_class>[0-9]{2})_(?P<sub_class>[A-Z])_(?P<count>[0-9]+).txt')


def get_clf_from_file_name(file_name: str) -> str:
    """
    judge and get classification str from the classification json docs' filename, if the file name not match
    the format such as 'A_01_H_300.txt' or there is no json in the file return None.
    :param file_name:
    :return:
    """
    matcher = pattern.match(file_name)
    if matcher:
        count = matcher.group('count')
        if count == '0':
            return None
        section = matcher.group('section')
        main_class = matcher.group('main_class')
        sub_class = matcher.group('sub_class')

        return '%s_%s_%s' % (section, main_class, sub_class)
    else:
        return None


def get_json(file_name: str) -> list:
    """
    Read doc json from file line by line, but only retain the followed three fields: pubId, title, abs.
    Collect the jsons to a list and return it.
    :param file_name:
    :return: list of json obj
    """
    json_list = []
    with open(file_name, encoding='utf-8') as f:
        for line in f.readlines():
            origin_json = json.loads(line)
            modified_json = {'id':    origin_json['pubId'],
                             'title': origin_json['title'],
                             'abs':   origin_json['abs']}
            json_list.append(modified_json)
    return json_list


def get_keywords_text_rank(data, top_k) -> pd.DataFrame:
    """
    get key words by text rank
    :param data:
    :param top_k:
    :return:
    """
    id_list, title_list, abstract_list = data['id'], data['title'], data['abs']
    ids, titles, keys = [], [], []
    # load custom stopwords
    jieba.analyse.set_stop_words("data/stopWord.txt")
    for index in range(len(id_list)):
        # concat title and abstract
        text = '%s。%s' % (title_list[index], abstract_list[index])
        print('process {} by text rank:'.format(title_list[index]))
        keywords = jieba.analyse.textrank(text, topK=top_k, allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))
        word_split = ' '.join(keywords)
        print('keywords is {}'.format(word_split))
        keys.append(word_split.encode('utf-8'))
        ids.append(id_list[index])
        titles.append(title_list[index])

    result = pd.DataFrame({"id": ids, "key": keys}, columns=['id', 'key'])
    return result


def main():
    work_dir = '/home/tqhy/ip_nlp/resources/data'
    results = batch(work_dir)
    for clf_str, result in results:
        result_file = 'result/keys_TextRank/%s.json' % (clf_str)
        result.to_json(result_file, force_ascii=False)


if __name__ == '__main__':
    main()
