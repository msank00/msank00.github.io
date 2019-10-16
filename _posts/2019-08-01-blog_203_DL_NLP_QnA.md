---
layout: post
title:  "Blog 203: Deep Learning: Natural Language Processing QnA"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

## **How do you find the similar documents related to some query sentence/search?**

+ Simplest apporach is to do tf-idf of both documents and query, and then measure cosine distance (i.e., dot product)
+ On top of that, if you use `SVD/PCA/LSA` on the tfidf matrix, it should further improve results. 

#### Source:

- [Blog1](https://www.r-bloggers.com/build-a-search-engine-in-20-minutes-or-less/)
- [Imp Blog2](http://searchivarius.org/blog/brief-overview-querysentence-similarity-functions)
----

## What is POS tagger? 

 > A Part-Of-Speech Tagger (POS Tagger) is a piece of software that reads text in some language and assigns parts of speech to each word (and other token), such as noun, verb, adjective, etc.

PoS taggers use an algorithm to label terms in text bodies. These taggers make more complex categories than those defined as basic PoS, with tags such as “noun-plural” or even more complex labels.

### How to build a POS simple tagger? How to account for the new word?

**Simple Idea:** 

- First collect tagged sentences

```py
import nltk
tagged_sentences = nltk.corpus.treebank.tagged_sents()
```

- Preprocess the sentences and create `[(word_1, tag_1), ... (word_n, tag_n)]`. This becomes your $X$ and $Y$.

- Train a multiclass classification algorithm like RandomForest and build your model

- Give test sentence, split into words, feed to the model and get corresponding tags.

**Resource:**
- [Build your own POS tagger](https://nlpforhackers.io/training-pos-tagger/)

- [Build more complex POS tagger with Keras](https://nlpforhackers.io/lstm-pos-tagger-keras/)
- [NLP for Hackers](https://nlpforhackers.io)

----
## How would you train a model that identifies whether the word “Apple” in a sentence belongs to the fruit or the company?

- This is a classic example of `Named Entity Recognition`. It is a statistical technique that (most commonly) uses `Conditional Random Fields` to find named entities, based on having been trained to learn things about named entities. Essentially, it looks at the content and context of the word, (looking back and forward a few words), to estimate the probability that the word is a named entity. 


### How to build your own NER model?

- It's a supervised learning problem. So first you need to get labelled data, i.e `words` and `entity_tag`  pair. For example (`London`,`GEO`), (`Apple Corp.`, `ORG`) and then train some model.
- As a novice model, apply scikit learn multiclass classification algorithm.
- For a more mature model use scikit learn `conditional random field` technique for creating a better model.

#### Resource:

- [named-entity-recognition-and-classification-with-scikit-learn](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2)
- [training ner with sklearn](https://nlpforhackers.io/training-ner-large-dataset/)
- [deep learning based NER](https://appliedmachinelearning.blog/2019/04/01/training-deep-learning-based-named-entity-recognition-from-scratch-disease-extraction-hackathon/)
- [Named Entity Recognition (NER) with keras and tensorflow](https://towardsdatascience.com/named-entity-recognition-ner-meeting-industrys-requirement-by-applying-state-of-the-art-deep-698d2b3b4ede)

### What is LBFGS algorithm?

- In numerical optimization, the `Broyden–Fletcher–Goldfarb–Shanno` (BFGS) algorithm is an `iterative method` for solving `unconstrained nonlinear optimization` problems.
- The BFGS method belongs to quasi-Newton methods, a class of hill-climbing optimization techniques that seek a stationary point of a (preferably twice continuously differentiable) function. For such problems, a necessary condition for optimality is that the gradient be zero. Newton's method and the BFGS methods are not guaranteed to converge unless the function has a quadratic Taylor expansion near an optimum. However, BFGS can have acceptable performance even for non-smooth optimization instances.

**Resource:**

- [Wikipedia](https://www.google.com/search?q=BFGS+algorithm&ie=utf-8&oe=utf-8&client=firefox-b-e&safe=strict)

----

## How would you find all the occurrences of quoted text in a news article?

As Mayur mentioned, you can do a regex to pick up everything between quotes

```py
list = re.findall("\".*?\"", string)
```

The problem you'll run into is that there can be a surprisingly large amount of things between quotation marks that are actually not quotations.

```py
"(said|writes|argues|concludes)(,)? \".?\""
```

**Resource:**
- [StackOverflow](https://stackoverflow.com/questions/37936461/how-to-extract-quotations-from-text-using-nltk)

---
## **Latent Semantic Analysis (LSA) for Text Classification Tutorial**?

### What is Latent Semantic Indexing?

Latent semantic indexing is a mathematical technique to extract information from unstructured data. It is based on the principle that words used in the same context carry the same meaning.

In order to identify relevant (concept) components, or in other words, aims to group words into classes that represent concepts or semantic fields, this method applies `Singular Value Decomposition` to the `Term-Document matrix` **tf-idf**. As the name suggests this matrix consists of words as rows and document as columns. 


- [Blog](https://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/)

----
## **Explain TF-IDF ? What is the drawback of Tf-Idf ? How do you overcome it ?**

### **Advantages:**
- Easy to compute
- You have some basic metric to extract the most descriptive terms in a document
- You can easily compute the similarity between 2 documents using it

### **Disadvantages:**
- TF-IDF is based on the bag-of-words (BoW) model, therefore it does not capture position in text, semantics, co-occurrences in different documents, etc.
- For this reason, TF-IDF is only useful as a lexical level feature
- Cannot capture semantics (e.g. as compared to topic models, word embeddings)


- [link](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

## **What is word2vec? What is the cost function for skip-gram model(k-negative sampling)?**

+ [cs224-lecture](https://www.youtube.com/watch?v=ASn7ExxLZws&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=3)
+ [keras implementation](http://adventuresinmachinelearning.com/word2vec-keras-tutorial/)
+ [AVB-Different word counting technique](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)

## So As per my experience, Tf-Idf fails in document classification/clustering ? How can you improve further ?
---

## **What are word2vec vectors?**

`Word2Vec`  embeds words in a lower-dimensional vector space using a shallow neural network. The result is a set of word-vectors where vectors close together in vector space have similar meanings based on context, and word-vectors distant to each other have differing meanings. For example, `apple` and `orange` would be close together and apple and gravity would be relatively far. There are two versions of this model based on skip-grams (SG) and continuous-bag-of-words (CBOW).

---

## **How can I design a chatbot ? (I had little idea but I tried answering it with intent and response tf-idf based similarity)**
+ [Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me)
 
## **Can I develop a chatbot with RNN providing a intent and response pair in input ?**
## Suppose I developed a chatbot with RNN/LSTMs on Reddit dataset. 
   It gives me 10 probable responses ? 
   How can I choose the best reply Or how can I eliminate others replies ?**

1. How do you perform text classification ?
2. How can you make sure to learn a context !! Well its not possible with TF-IDF ? 
	+ I told him about taking n-grams say n = 1, 2, 3, 4 and concatenating tf-idf of them to make a long count vector ?
Okay that is the baseline people start with ? What can you do more with machine learning ? 
(I tried suggesting LSTM with word2vec or 1D-CNN with word2vec for classification but 
 he wanted improvements in machine learning based methods :-|)
10. **How does neural networks learns non-linear shapes when it is made of linear nodes ? What makes it learn non-linear boundaries ?**
------
1. **What is the range of sigmoid function** ?
2. 
3. Text classification method. How will you do it ?
4. Explain Tf-Idf ? 
5. What are bigrams & Tri-grams ? Explain with example of Tf-Idf of bi-grams & trigrams with a text sentence.
6. **What is an application of word2vec** ? Example.
7. **How will you design a neural network ?** How about making it very deep ? Very basic questions on neural network.?
8.  <span style="color:red">**How does LSTM work ? How can it remember the context ?**</span>
  + **Must Watch:** CS231n by Karpathy in 2016 course and Justin in 2017 course.
9.  How did you perform language identification ? What were the  feature ?
10. How did you model classifiers like speech vs music and speech vs non-speech ?
11. How can deep neural network be applied in these speech analytics applications ?

----

## **Role-specific questions**

**Natural language processing**

1. [x] What is part of speech (POS) tagging? What is the simplest approach to building a POS tagger that you can imagine?
2. [x] How would you build a POS tagger from scratch given a corpus of annotated sentences? How would you deal with unknown words?
3. [x] How would you train a model that identifies whether the word “Apple” in a sentence belongs to the fruit or the company?
4. How would you find all the occurrences of quoted text in a news article?
5. How would you build a system that auto corrects text that has been generated by a speech recognition system?
6. What is latent semantic indexing and where can it be applied?
7. How would you build a system to translate English text to Greek and vice-versa?
8. How would you build a system that automatically groups news articles by subject?
9. What are stop words? Describe an application in which stop words should be removed.
10. How would you design a model to predict whether a movie review was positive or negative?
11. Do you know about latent semantic indexing? Where can you apply it?
12. Is it possible to find all the occurrences of quoted text in an article? If yes, explain how?
13. What is a POS tagger? Explain the simplest approach to build a POS tagger?
14. Which is a better algorithm for POS tagging – SVM or hidden Markov models?
15. What is the difference between shallow parsing and dependency parsing?
16. What package are you aware of in python which is used in NLP and ML?
17. Explain one application in which stop words should be removed.
18. How will you train a model to identify whether the word “Raymond” in a sentence represents a person’s name or a company?
19. Which is better to use while extracting features character n-grams or word n-grams? Why?
20. What is a POS tagger? How can you built one?
21. What is dimensionality reduction?
22. Explain the working of SVM/NN/Maxent algorithms
23. Which is a better algorithm for POS tagging - SVM or hidden markov models ? why?
24. What packages are you aware of in python which are used in NLP and ML?
25. What are conditional random fields ?
26. When can you use Naive Bayes algorithm for training, what are its advantages and disadvantages?
27. How would you build a POS tagger from scratch given a corpus of annotated sentences? How would you deal with unknown words?
28. What is part of speech (POS) tagging? What is the simplest approach to building a POS tagger that you can imagine?




**Related fields such as information theory, linguistics and information retrieval**

1. What is entropy? How would you estimate the entropy of the English language?
2. What is a regular grammar? Does this differ in power to a regular expression and if so, in what way?
3. What is the TF-IDF score of a word and in what context is this useful?
4. How does the PageRank algorithm work?
5. What is dependency parsing?
6. What are the difficulties in building and using an annotated corpus of text such as the Brown Corpus and what can be done to mitigate them?
7. Differentiate regular grammar and regular expression.
8. How will you estimate the entropy of the English language?
9. Describe dependency parsing?
10. What do you mean by Information rate?
11. Explain Discrete Memoryless Channel (DMC).
12. How does correlation work in text mining?
13. How to calculate TF*IDF for a single new document to be classified?
14. How to build ontologies?
15. What is an N-gram in the context of text mining?
16. What do you know about linguistic resources such as WordNet?
17. Explain the tools you have used for training NLP models?


**Tools and languages**

1. What tools for training NLP models (nltk, Apache OpenNLP, GATE, MALLET etc…) have you used?
2. Do you have any experience in building ontologies?
3. Are you familiar with WordNet or other related linguistic resources?
4. Do you speak any foreign languages?

----

***Question Source**

- [MLInterview](https://github.com/theainerd/MLInterview)
- [Data-Science-Interview-Resources](https://github.com/rbhatia46/Data-Science-Interview-Resources)
- [NoML](https://weifoo.gitbooks.io/noml/content/)
- 