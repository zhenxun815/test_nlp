## Create Corpus

**_Corpus_**_(a Bag of Words)_, it is a object that contains the _**word id**_ and its 
**_frequency_** in **_each document_**.We can think of it as gensim’s equivalent of 
a **_[Document-Term matrix](https://en.wikipedia.org/wiki/Document-term_matrix)_**.

Once we have the updated dictionary, all we need to do to create a bag of words corpus 
is to pass the tokenized list of words to the `Dictionary.doc2bow()`
---
##### 1. Core Objects and Functions
```
gensim.corpora.Dictionary.doc2bow(self, document, allow_update=False, return_missing=False)

Convert `document` into the bag-of-words (BoW) format = list of (token_id, token_count).

Parameters
----------
document :  list of str
    Input document.
allow_update : bool, optional
    If True - update dictionary in the process (i.e. add new tokens and update frequencies).
return_missing : bool, optional
    Also return missing tokens (that doesn't contains in current dictionary).

Return
------
list of (int, int)
    BoW representation of `document`
list of (int, int), dict of (str, int)
    If `return_missing` is True, return BoW representation of `document` + dictionary with 
    missing tokens and their frequencies.

Examples
--------
>>> from gensim.corpora import Dictionary
>>> dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
>>> dct.doc2bow(["this","is","máma"])
[(2, 1)]
>>> dct.doc2bow(["this","is","máma"], return_missing=True)
([(2, 1)], {u'this': 1, u'is': 1})

```
#### 2.Example of Creating from List of Sentences
```python

from test_gensim.dictionary.gen_token_list import *
from gensim.corpora import Dictionary

def create_corpus(docs, my_dict):
    token_list = docs_list2token_list(docs)

    print('before call doc2bow, dictionary is {}'.format(my_dict.token2id))
    corpus = [my_dict.doc2bow(doc, allow_update=True) for doc in token_list]
    print('after call doc2bow, dictionary is {}'.format(my_dict.token2id))
    return corpus

# test block
if __name__ == '__main__':
    dictionary = Dictionary()
    documents1 = ['一种 大头菜 自然风', '风 主要 包括 大头菜 风', '架 主要 包括 底座 支柱']
    corpus = create_corpus(documents1, dictionary)
    print('corpus is {}'.format(corpus))
    
    documents2 = ['一种 海绵拔 轮', '支架 上 设有 海绵 辊轴', '同步 皮带轮 二 海绵 辊轴']
    corpus = create_corpus(documents2, dictionary)
    print('corpus is {}'.format(corpus))

```

out put is:
```
before call doc2bow, dictionary is {}
after call doc2bow, dictionary is {'一种': 0, '大头菜': 1, '自然风': 2, '主要': 3, '包括': 4, 
'风': 5, '底座': 6, '支柱': 7, '架': 8}
corpus is [
[(0, 1), (1, 1), (2, 1)], 
[(1, 1), (3, 1), (4, 1), (5, 2)], 
[(3, 1), (4, 1), (6, 1), (7, 1), (8, 1)]
]


before call doc2bow, dictionary is {'一种': 0, '大头菜': 1, '自然风': 2, '主要': 3, '包括': 4, 
'风': 5, '底座': 6, '支柱': 7, '架': 8}
after call doc2bow, dictionary is {'一种': 0, '大头菜': 1, '自然风': 2, '主要': 3, '包括': 4, 
'风': 5, '底座': 6, '支柱': 7, '架': 8, '海绵拔': 9, '轮': 10, '上': 11, '支架': 12, '海绵': 13, 
'设有': 14, '辊轴': 15, '二': 16, '同步': 17, '皮带轮': 18}
corpus is [
[(0, 1), (9, 1), (10, 1)], 
[(11, 1), (12, 1), (13, 1), (14, 1), (15, 1)], 
[(13, 1), (15, 1), (16, 1), (17, 1), (18, 1)]
]
```

The (0, 1) in the line 1 of first paragraph means, the word,'一种', with id 0, appears once 
in the 1st document: '一种 大头菜 自然风'. 

Likewise, the (5, 2) in the second list item means 
the word, '风', with id 5, appears twice in the second document, '风 主要 包括 大头菜 风'. And so on.

Note that we set the param `allow_update` to `True`, so in the second out put paragraph, the dictionary 
add new tokens of documents2.

In order to make the output human readable, we can use the dictionary to do a conversion:
```python

def print_corpus_human_readable(dictionary, corpus):
    word_counts = [[(dictionary[token_id], count) for token_id, count in item] for item in corpus]
    print('corpus is {}'.format(word_counts))
``` 
Now, we can see the out put like this:

```
[
[('一种', 1), ('海绵拔', 1), ('轮', 1)], 
[('上', 1), ('支架', 1), ('海绵', 1), ('设有', 1), ('辊轴', 1)], 
[('海绵', 1), ('辊轴', 1), ('二', 1), ('同步', 1), ('皮带轮', 1)]
]
```

#### 3. Example of Creating from Files

```python
import os

from gensim.corpora import Dictionary
from test_gensim.dictionary.read_files import preprocess_file


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
            file_tokens = preprocess_file(file_path)
            yield (file_name, self.dictionary.doc2bow(file_tokens))


def get_dictionary_of_dir(dir_path):
    """
    Generate a Dictionary of text files under the given dir
    :rtype: Dictionary
    :param dir_path:
    :return:
    """
    dictionary = Dictionary()
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('seg'):
            continue

        file_path = os.path.join(dir_path, file_name)
        file_tokens = [preprocess_file(file_path)]
        dictionary.add_documents(file_tokens)

    return dictionary

# test block
if __name__ == '__main__':
    dir_dictionary = get_dictionary_of_dir('../resources')
    for file_corpus in BoWCorpus(dir_dictionary, '../resources'):
        file_name = file_corpus[0]
        corpus = [(dir_dictionary[index], count) for index, count in file_corpus[1]]
        print('file_name is: {}'.format(file_name))
        print('corpus is {}'.format(corpus))
```

output:

```
file_name is: CN104188073B.seg
corpus is [('一体', 5), ('一定', 2), ('上下', 1)......('雨淋', 1), ('需要', 3), ('食品', 1), ('食用', 2)]
file_name is: CN105253527A.seg
corpus is [('之间', 3), ('包括', 1), ('固定', 1)...... ('轴承', 2), ('辊轴', 5), ('过度', 2)]

```