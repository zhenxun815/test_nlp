## TF-IDF 

#### 1. Core Object
```
gensim.models.TfidfModel(self, corpus=None, id2word=None, dictionary=None, wlocal=utils.identity, 
wglobal=df2idf, normalize=True, smartirs=None)  

Compute tf-idf by multiplying a local component (term frequency) with a global component
        (inverse document frequency), and normalizing the resulting documents to unit length.
        Formula for non-normalized weight of term :math:`i` in document :math:`j` in a corpus of :math:`D` documents

        .. math:: weight_{i,j} = frequency_{i,j} * log_2 \\frac{D}{document\_freq_{i}}

        or, more generally

        .. math:: weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document\_freq_{i}, D)

        so you can plug in your own custom :math:`wlocal` and :math:`wglobal` functions.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int), optional
            Input corpus
        id2word : {dict, :class:`~gensim.corpora.Dictionary`}, optional
            Mapping token - id, that was used for converting input data to bag of words format.
        dictionary : :class:`~gensim.corpora.Dictionary`
            If `dictionary` is specified, it must be a `corpora.Dictionary` object and it will be used.
            to directly construct the inverse document frequency mapping (then `corpus`, if specified, is ignored).
        wlocals : function, optional
            Function for local weighting, default for `wlocal` is :func:`~gensim.utils.identity`
            (other options: :func:`math.sqrt`, :func:`math.log1p`, etc).
        wglobal : function, optional
            Function for global weighting, default is :func:`~gensim.models.tfidfmodel.df2idf`.
        normalize : bool, optional
            It dictates how the final transformed vectors will be normalized. `normalize=True` means set to unit length
            (default); `False` means don't normalize. You can also set `normalize` to your own function that accepts
            and returns a sparse vector.
        smartirs : str, optional
            SMART (System for the Mechanical Analysis and Retrieval of Text) Information Retrieval System,
            a mnemonic scheme for denoting tf-idf weighting variants in the vector space model.
            The mnemonic for representing a combination of weights takes the form XYZ,
            for example 'ntc', 'bpn' and so on, where the letters represents the term weighting of the document vector.

            Term frequency weighing:
                * `n` - natural,
                * `l` - logarithm,
                * `a` - augmented,
                * `b` - boolean,
                * `L` - log average.

            Document frequency weighting:
                * `n` - none,
                * `t` - idf,
                * `p` - prob idf.

            Document normalization:
                * `n` - none,
                * `c` - cosine.

```

#### 2. Example
```python
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import numpy as np

if __name__ == '__main__':
    docs = ['一种 大头菜 自然风 脱水 设备 其 特征 在于 所述 的 大头菜 自然风 脱水 设备 主要 包括 大头菜',
            '风 脱水 架 和 大头菜 风 脱水 网袋 所述 的 大头菜 风 脱水 架 主要 包括 底座 支柱 横架 横向',
            '连接 承重杆 所述 的 底座 通过 中间 的 多边形 孔 与 支柱 的 下端 的 多边形 柱 配合 而 固定']

    tokenized_docs = [simple_preprocess(doc, min_len=1) for doc in docs]

    my_dct = Dictionary(tokenized_docs)
    print('dictionary is {}'.format(my_dct.token2id) )

    corpus = [my_dct.doc2bow(doc) for doc in tokenized_docs]
    for index,bow in enumerate(corpus):
        bow = [[my_dct[index], count] for index, count in bow]
        print('bow of doc {} is'.format(index))
        print(bow)

    tf_idf_model = TfidfModel(corpus, id2word=my_dct, dictionary=my_dct, smartirs='ntc')
    for index, doc in enumerate(tf_idf_model[corpus]):
        tf_idf_info = [[my_dct[id], np.around(freq, decimals=2)] for id, freq in doc]
        print('tf-idf model info of doc {} is:'.format(index))
        print(tf_idf_info)
```

output:  
```
dictionary is:
    {'一种': 0, '主要': 1, '其': 2, '包括': 3, '在于': 4, '大头菜': 5, '所述': 6, 
    '特征': 7, '的': 8, '脱水': 9, '自然风': 10, '设备': 11, '和': 12, '底座': 13, '支柱': 14, '架': 15, 
    '横向': 16, '横架': 17, '网袋': 18, '风': 19, '下端': 20, '与': 21, '中间': 22, '固定': 23, 
    '多边形': 24, '孔': 25, '承重杆': 26, '柱': 27, '而': 28, '连接': 29, '通过': 30, '配合': 31}

bow of doc 0 is
    [['一种', 1], ['主要', 1], ['其', 1], ['包括', 1], ['在于', 1], ['大头菜', 3], ['所述', 1], 
    ['特征', 1], ['的', 1], ['脱水', 2], ['自然风', 2], ['设备', 2]]
bow of doc 1 is
    [['主要', 1], ['包括', 1], ['大头菜', 2], ['所述', 1], ['的', 1], ['脱水', 3], ['和', 1], 
    ['底座', 1], ['支柱', 1], ['架', 2], ['横向', 1], ['横架', 1], ['网袋', 1], ['风', 3]]
bow of doc 2 is
    [['所述', 1], ['的', 4], ['底座', 1], ['支柱', 1], ['下端', 1], ['与', 1], ['中间', 1], ['固定', 1], 
    ['多边形', 2], ['孔', 1], ['承重杆', 1], ['柱', 1], ['而', 1], ['连接', 1], ['通过', 1], ['配合', 1]]

tf-idf model info of doc 0 is:
    [['一种', 0.27], ['主要', 0.1], ['其', 0.27], ['包括', 0.1], ['在于', 0.27], ['大头菜', 0.3], 
    ['特征', 0.27], ['脱水', 0.2], ['自然风', 0.53], ['设备', 0.53]]
tf-idf model info of doc 1 is:
    [['主要', 0.08], ['包括', 0.08], ['大头菜', 0.17], ['脱水', 0.25], ['和', 0.23], ['底座', 0.08], 
    ['支柱', 0.08], ['架', 0.46], ['横向', 0.23], ['横架', 0.23], ['网袋', 0.23], ['风', 0.68]]
tf-idf model info of doc 2 is:
    [['底座', 0.09], ['支柱', 0.09], ['下端', 0.26], ['与', 0.26], ['中间', 0.26], ['固定', 0.26], 
    ['多边形', 0.51], ['孔', 0.26], ['承重杆', 0.26], ['柱', 0.26], ['而', 0.26], ['连接', 0.26], 
    ['通过', 0.26], ['配合', 0.26]]

```