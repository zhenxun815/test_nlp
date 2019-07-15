# !/usr/bin/python
# coding=utf-8
# 采用Word2Vec词聚类方法抽取关键词2——根据候选关键词的词向量进行聚类分析
import sys, os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from pprint import pprint


def gen_data(path):
    titlelist = []
    abslist = []
    file = open(path, 'rb')
    for line in file.readlines():
        dic = json.loads(line)
        titlelist.append(dic['title'])
        abslist.append(dic['abs'])

    sumlist = {
        "title": titlelist,
        "abstract": abslist}
    data = pd.DataFrame(sumlist)
    shape = data.shape[0]
    idlist = list(range(1, shape + 1))
    data['id'] = idlist
    return data


# 对词向量采用K-means聚类抽取TopK关键词
def getkeywords_kmeans(data, topK):
    words = data["word"]  # 词汇
    vecs = data.iloc[:, :-1]  # 向量表示
    # print('vecs is {}'.format(vecs))
    kmeans = KMeans(n_clusters=1, random_state=10).fit(vecs)
    labels = kmeans.labels_  # 类别结果标签
    labels = pd.DataFrame(labels, columns=['label'])
    new_df = pd.concat([labels, vecs], axis=1)
    df_count_type = new_df.groupby('label').size()  # 各类别统计个数
    print(df_count_type)
    vec_center = kmeans.cluster_centers_  # 聚类中心

    # 计算距离（相似性） 采用欧几里得距离（欧式距离）
    distances = []
    vec_words = np.array(vecs)  # 候选关键词向量，dataFrame转array
    vec_center = vec_center[0]  # 第一个类别聚类中心,本例只有一个类别
    length = len(vec_center)  # 向量维度
    for index in range(len(vec_words)):  # 候选关键词个数
        cur_wordvec = vec_words[index]  # 当前词语的词向量
        dis = 0  # 向量距离
        for index2 in range(length):
            dis += (vec_center[index2] - cur_wordvec[index2]) * (vec_center[index2] - cur_wordvec[index2])
        dis = math.sqrt(dis)
        distances.append(dis)
    distances = pd.DataFrame(distances, columns=['dis'])

    result = pd.concat([words, labels, distances], axis=1)  # 拼接词语与其对应中心点的距离
    result = result.sort_values(by="dis", ascending=True)  # 按照距离大小进行升序排序

    # 将用于聚类的数据的特征维度降到2维
    # pca = PCA(n_components=2)
    # new_pca = pd.DataFrame(pca.fit_transform(new_df))
    # print new_pca
    # 可视化
    # d = new_pca[new_df['label'] == 0]
    # plt.plot(d[0],d[1],'r.')
    # d = new_pca[new_df['label'] == 1]
    # plt.plot(d[0], d[1], 'go')
    # d = new_pca[new_df['label'] == 2]
    # plt.plot(d[0], d[1], 'b*')
    # # plt.gcf().savefig('kmeans.png')
    # plt.show()

    # 抽取排名前topK个词语作为文本关键词
    wordlist = np.array(result['word'])  # 选择词汇列并转成数组格式
    last_index = topK if len(wordlist) > topK else len(wordlist)
    word_split = [wordlist[x] for x in range(0, last_index)]  # 抽取前topK个词汇
    word_split = " ".join(word_split)
    return word_split


def main():
    # 读取数据集
    dataFile = 'data/raw_docs.json'
    articleData = pd.read_json(dataFile, encoding='utf-8', lines=True)
    ids, titles, keys = [], [], []

    rootdir = "result/vecs"  # 词向量文件根目录
    fileList = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    # 遍历文件
    task_count = len(fileList)
    for i in range(task_count):
        print('task remain {}'.format(task_count - i))
        filename = fileList[i]
        path = os.path.join(rootdir, filename)
        print('vec file path is {}'.format(path))
        if os.path.isfile(path) and path.endswith('json'):
            data = pd.read_json(path, encoding='utf-8', lines=False)  # 读取词向量文件数据
            # print('frame data is {}'.format(data))
            artile_keys = getkeywords_kmeans(data, 10)  # 聚类算法得到当前文件的关键词
            # 根据文件名获得文章id以及标题
            (shortname, extension) = os.path.splitext(filename)  # 得到文件名和文件扩展名
            t = shortname.split("_")
            article_id = int(t[len(t) - 1])  # 获得文章id
            artile_tit = articleData.iloc[article_id - 1]['title']  # 获得文章标题
            # artile_tit = list(artile_tit)[0]  # series转成字符串
            ids.append(article_id)
            titles.append(artile_tit)
            keys.append(artile_keys.encode("utf-8"))

    # 所有结果写入文件
    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
    result = result.sort_values(by="id", ascending=True)  # 排序
    result.to_json("result/keys_word2vec.json", force_ascii=False)
    print('all tasks compelte...')


if __name__ == '__main__':
    main()
