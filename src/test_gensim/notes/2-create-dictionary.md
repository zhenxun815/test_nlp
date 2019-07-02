## Create Dictionary

```
gensim.corpora.Dictionary

Parameters
----------
documents : iterable of iterable of str, optional
    Documents that used for initialization.
prune_at : int, optional
    Total number of unique words. Dictionary will keep not more than 
    `prune_at` words.

Examples
--------
>>> from gensim.corpora import Dictionary
>>>
>>> texts = [['human', 'interface', 'computer']]
>>> dct = Dictionary(texts)  # fit dictionary
>>> dct.add_documents([["cat", "say", "meow"], ["dog"]])  
>>> # update dictionary with new documents
>>> dct.doc2bow(["dog", "computer", "non_existent_word"])
[(0, 1), (6, 1)]
```

```
gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)

Convert a document into a list of tokens (also with lowercase 
and optional de-accents),
used :func:`~gensim.utils.tokenize`.

Parameters
----------
doc : str
    Input document.
deacc : bool, optional
    If True - remove accentuation from string by :func:`~gensim.utils.deaccent`.
min_len : int, optional
    Minimal length of token in result (inclusive).
max_len : int, optional
    Maximal length of token in result (inclusive).

Returns
-------
list of str
    Tokens extracted from `doc`.

```
 
#### 1.Create from Lists of Sentences
```python
from gensim import corpora
from gensim.utils import simple_preprocess


def docs_list2token_list(documents):
    # return [[text for text in doc.split()] for doc in documents]
    return [simple_preprocess(doc, min_len=1) for doc in documents]
  
    
documents1 = ['一种 大头菜 自然风','主要 包括 大头菜 风','架 主要 包括 底座 支柱']
documents2 = ['一种 海绵拔 轮','支架 上 设有 海绵 辊轴','同步 皮带轮 二 海绵 辊轴']

texts1 = docs_list2token_list(documents1)
texts2 = docs_list2token_list(documents2)


dictionary = corpora.Dictionary(texts1)
print('after add texts1, dictionary is {}'.format(dictionary.token2id))
"""
{'一种': 0, '大头菜': 1, '自然风': 2, '主要': 3, '包括': 4, 
'风': 5, '底座': 6, '支柱': 7, '架': 8}
"""

dictionary.add_documents(texts2)
print('after add texts2, dictionary is {}'.format(dictionary.token2id))
"""
{'一种': 0, '大头菜': 1, '自然风': 2, '主要': 3, '包括': 4, 
'风': 5, '底座': 6, '支柱': 7, '架': 8, 
'海绵拔': 9, '轮': 10, '上': 11, '支架': 12, '海绵': 13, 
'设有': 14, '辊轴': 15, '二': 16, '同步': 17, '皮带轮': 18}
"""

```

#### 2.Create from File of Tokens
  Read a file line-by-line and use gensim’s simple_preprocess to process 
  one line of the file at a time.
  The advantage here is it let us read an entire text file without loading 
  the file in memory all at once.


```python
from gensim import corpora
from gensim.utils import simple_preprocess


def gen_texts_list_from_file(tokens_file):
    with open(tokens_file, 'r', encoding='utf-8') as f:
        return [simple_preprocess(line, min_len=1) for line in f]
        

texts_list = gen_texts_list_from_file('../resources/CN105253527A.seg')
dict_from_file = corpora.Dictionary(texts_list)
print('dict is {}'.format(dict_from_file.token2id))
"""
{'一': 0, '上': 1, '其': 2, '包括': 3, '同步': 4, '和': 5, '在于': 6, '拔辊': 7, 
'支架': 8, '机构': 9, '海绵拔': 10, '特征': 11, '电机': 12, '种': 13, '设有': 14, 
......
'连接': 36, '述': 37, '之间': 38, '座': 39, '海绵辊': 40, '螺纹': 41, '调节': 42, 
'与': 43, '滚子': 44}
"""
```

#### 3.Create from Files of Tokens

```python
import os
from gensim import corpora
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
                

dict_from_files = corpora.Dictionary(ReadTxtUnderDir('../resources/'))
print('dic from files is {}'.format(dict_from_files))
"""
{'主要': 0, '包括': 1, '在于': 2, '大头菜': 3, '所述': 4, '特征': 5, '脱水': 6, 
... ...
'辊轴': 242, '过度': 243, '带座': 244, '螺母': 245, '设在': 246, '轴承': 247, 
'皮带': 248, '海绵辊': 249, '螺纹': 250, '调节': 251, '滚子': 252}
"""


```