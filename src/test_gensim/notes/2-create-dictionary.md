## Create Dictionary

```
gensim.corpora.Dictionary

Parameters
----------
documents : iterable of iterable of str, optional
    Documents that used for initialization.
prune_at : int, optional
    Total number of unique words. Dictionary will keep not more than 
    `prune_at` words, default value is 2000000.

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


dictionary.add_documents(texts2)
print('after add texts2, dictionary is {}'.format(dictionary.token2id))


```

output:
```
after add texts1, dictionary is {'一种': 0, '大头菜': 1, '自然风': 2 ...... '风': 5, '底座': 6, '支柱': 7, '架': 8}
after add texts2, dictionary is {'一种': 0, '大头菜': 1, '自然风': 2 ...... '二': 16, '同步': 17, '皮带轮': 18}
```

#### 2.Create from File of Tokens
  Read a file line-by-line and use gensim’s simple_preprocess to process 
  one line of the file at a time.
  The advantage here is it let us read an entire text file without loading 
  the file in memory all at once.


```python
from gensim.utils import simple_preprocess


def tokens_file2token_list(tokens_file, min_len=1):
    with open(tokens_file, 'r', encoding='utf-8') as f:
        return [simple_preprocess(line, min_len=min_len) for line in f]
        
```

test code:
```python
from gensim import corpora
from test_gensim.dictionary.gen_token_list import tokens_file2token_list

texts_list = tokens_file2token_list('../resources/CN105253527A.seg')
dict_from_file = corpora.Dictionary(texts_list)
print('dict is {}'.format(dict_from_file.token2id))

```

output:
```
dict is {'一': 0, '上': 1, '其': 2, '包括': 3, ...... '与': 43, '滚子': 44}
```

#### 3.Create from Files of Tokens

```python
import os
from gensim.utils import simple_preprocess


class ReadTxt:
    """
    Read file line by line and parse each line to a tokens list.
    Note that the generated list is a 2D list, each element list in it
    consist of the tokens of each line.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        for line in open(self.file_path, encoding='utf-8'):
            yield simple_preprocess(line, min_len=2)


def preprocess_dir(dir_path):
    """
    Get tokens of all file under the given dir path
    :param dir_path:
    :return: dict, item key is file path, value is list of tokens
    """
    file_tokens = dict()
    for file_name in os.listdir(dir_path):
        if not file_name.endswith('seg'):
            continue
        file_path = os.path.join(dir_path, file_name)
        file_tokens_list = preprocess_file(file_path)
        file_tokens[file_path] = file_tokens_list
    return file_tokens


def preprocess_file(file_path):
    """
    Get list of tokens of the given file
    :param file_path: file to be processed
    :return: list of tokens
    """
    line_tokens_lists = ReadTxt(file_path)
    # flatten the 2D tokens list
    return [token for tokens in line_tokens_lists for token in tokens]

```

test code:
```python
from gensim import corpora
from test_gensim.dictionary.read_files import preprocess_dir

dir_tokens = preprocess_dir('../resources/')
my_dict = corpora.Dictionary()
for file_path, file_tokens in dir_tokens.items():
    print('file is {}, tokens is {}'.format(file_path,file_tokens))

    corpus = my_dict.doc2bow(file_tokens, allow_update=True)
    word_counts = [(my_dict[token_id], count) for token_id, count in corpus]
    print('corpus is {}'.format(word_counts))

```

output:
```
file is ../resources/CN104188073B.seg, tokens is ['大头菜', '自然风', ......'套在', '粘在', '网袋']
corpus is [('一体', 5), ('一定', 2), ('上下', 1) ...... ('需要', 3), ('食品', 1), ('食用', 2)]
file is ../resources/CN105253527A.seg, tokens is ['海绵拔', '机构', '特征', ...... '设有', '滚子', '轴承']
corpus is [('之间', 3), ('包括', 1), ('固定', 1) ...... ('调节', 1), ('轴承', 2), ('辊轴', 5), ('过度', 2)]
```