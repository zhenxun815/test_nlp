## Conceptions

> **_token_** _标记_:  
Typically means a ‘word’.

> **_document_** _文档_:  
Typically refer to a ‘sentence’ or ‘paragraph’

> **_Bag-of-words model_** _词袋模型_:  
In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, 
disregarding grammar and even word order but keeping multiplicity.

Example:

The following models a text document using bag-of-words. Here are two simple text documents:  

(1) `John likes to watch movies. Mary likes movies too.`   
(2) `John also likes to watch football games.`

Based on these two text documents, a list of words constructed as follows for each document:  

(1) `"John","likes","to","watch","movies","Mary","likes","movies","too"`  
(2) `"John","also","likes","to","watch","football","games"`

Representing each bag-of-words as a JSON object, and attributing to the respective JavaScript variable:  

BoW1 = `{"John":1,"likes":2,"to":1,"watch":1,"movies":2,"Mary":1,"too":1}`   
BoW2 = `{"John":1,"also":1,"likes":1,"to":1,"watch":1,"football":1,"games":1}`   
Each key is the word, and each value is the number of occurrences of that token in the given text document.


> **_corpus_** _语料库_:  
Typically a ‘collection of documents as a bag of words’. That is, for each document, a corpus contains 
each word’s id and its frequency count in that document.  As a result, information of the order of words is lost.

> **_dictionary_** _字典_:  
Typically used to create corpus, it maps each word to a unique id