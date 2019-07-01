#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_gensim.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/1/2019 15:36

from gensim import corpora


def trans_docs_to_array(documents):
    return [[text for text in doc.split()] for doc in documents]


def test_gen_dict():
    documents1 = ['一 种 大头菜 自然风 脱水 设备 其 特征 在于 所述 的 大头菜 自然风 脱水 设备',
                  '主要 包括 大头菜 风 脱水 架 和 大头菜 风 脱水 网袋  所述 的 大头菜 风 脱水',
                  '架 主要 包括 底座 支柱 横架 横向 连接 承重杆 所述 的 底座 通过 中间 的 ',
                  '多边形 孔 与 支柱 的 下端 的 多边形 柱 配合 而 固定 横 架 的 多边形 孔']

    documents2 = ['一 种 海绵拔 轮 机构 其 特征 在于 包括 拔辊 支架 和 同步 电机 支架 拔辊',
                  '支架 上 设有 海绵 辊轴 和 同步 皮带轮 一 同步 电机 支架 上 设有 过度 轴 和',
                  '同步 皮带轮 二 海绵 辊轴 由 螺母 进行 固定 过度 轴 设在 带座 轴承 上 所述',
                  '的 同步 皮带轮 一 和 同步 皮带轮 二 由 同步 皮带 连接 根据 权利 要求',
                  '所 述 的 一 种 海绵拔 轮 机构  其 特征 在于 所述 的 海绵 辊轴 之间 设有 海绵辊',
                  '海绵辊 和 海绵 辊轴 之间 设有 螺纹 调节 座 根据 权利 要求 所 述 的 一 种',
                  '海绵拔 轮 机构 其 特征 在于 所述 的 同步 皮带轮 一 与 海绵 辊轴 之间 设有 滚子 轴承']

    texts1 = trans_docs_to_array(documents1)
    dictionary = corpora.Dictionary(texts1)
    print(dictionary)
    print(dictionary.token2id)

    print('######################################################')
    texts2 = trans_docs_to_array(documents2)
    dictionary.add_documents(texts2)
    print(dictionary)
    print(dictionary.token2id)


test_gen_dict()
