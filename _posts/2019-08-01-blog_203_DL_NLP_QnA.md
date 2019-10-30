---
layout: post
title:  "Blog 203: Deep Learning: Natural Language Processing QnA"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

## For each Algo learn, 

- when to use and when not to use.
- Merits and Demerits
- Pros and Cons

## How to Prepare for ML-DL Interview

> Interviews are stressful, even more so when they are for your dream job. As someone who has been in your shoes, and might again be in your shoes in the future, I just want to tell you that it doesn’t have to be so bad. Each interview is a learning experience. An offer is great, but a rejection isn’t necessarily a bad thing, and is never the end of the world. I was pretty much rejected for every job when I began this process. Now I just get rejected at a less frequent rate. Keep on learning and improving. You’ve got this!

- [Tweet: Chip Huyen](https://twitter.com/chipro/status/1152077188985835521)
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

**Simple Solution:** you can do a regex to pick up everything between quotes

```py
list = re.findall("\".*?\"", string)
```

The problem you'll run into is that there can be a surprisingly large amount of things between quotation marks that are actually not quotations.

```py
"(said|writes|argues|concludes)(,)? \".?\""
```

But quotes are a tricky business. Lots of things look like quotes that aren't, and some things are more quote-like than others. The ideal approach would be able to account for some of that fuzziness in a way that pattern matching doesn't.

**Maximum Entropy Model:**

> This model considers all of the probability distributions that are `empirically consistent` with the training data; and chooses the distribution with the `highest entropy`.  A probability distribution is "empirically consistent" with a set of training data if its estimated frequency with which a class and a feature vector value co-occur is
equal to the actual frequency in the data.

- Many problems in natural language processing can be viewed as `linguistic classification` problems, in which `linguistic contexts` are used to predict `linguistic classes`. 
- **Maximum entropy models** offer a clean way to combine diverse pieces of contextual evidence in order to estimate the probability of a certain `linguistic class` occurring with a certain `linguistic context`.

In the above problem, use feature and apply maximum entropy model to classify if a paragraph has quotes or not. (For example, does the paragraph contain an attribution word like `“said?"`)


**Resource:**
- [StackOverflow](https://stackoverflow.com/questions/37936461/how-to-extract-quotations-from-text-using-nltk)
- [Using machine learning to extract quotes from text](https://www.revealnews.org/article/using-machine-learning-to-extract-quotes-from-text/)
- [A Simple Introduction to MaximumEntropy Models for NaturalLanguage Processing](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1083&context=ircs_reports)


---
## **Latent Semantic Analysis (LSA) for Text Classification Tutorial**?

- [AV](https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/)

### What is Latent Semantic Indexing?

- Latent semantic indexing is a mathematical technique to extract information from unstructured data. It is based on the principle that words used in the same context carry the same meaning.

> **Latent Semantic Analysis (LSA)** is a theory and method for extracting and representing the contextual-usage meaning of words by statistical computations applied to a large corpus of text.

- In order to identify relevant (concept) components, or in other words, aims to group words into classes that represent concepts or semantic fields, this method applies `Singular Value Decomposition` to the `Term-Document matrix` **tf-idf**. As the name suggests this matrix consists of words as rows and document as columns. 
- LSA itself is an unsupervised way of uncovering synonyms in a collection of documents.

### Where you can apply Latent Semantic Analysis?

- Text classification
- Topic Modelling

### Pros and Cons of LSA

`Latent Semantic Analysis` can be very useful as we saw above, but it does have its limitations. It’s important to understand both the sides of LSA so you have an idea of when to leverage it and when to try something else.

**Pros:**

- LSA is fast and easy to implement.
- It gives decent results, much better than a plain vector space model.

**Cons:**


- Since it is a linear model, it might not do well on datasets with non-linear dependencies.
- LSA assumes a Gaussian distribution of the terms in the documents, which may not be true for all problems.
- LSA involves SVD, which is computationally intensive and hard to update as new data comes up.


**Resource:**

- [Blog](https://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/)
- [Stepwise guide to Topic Modelling](https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/)
----

## How would you build a system that auto corrects text that has been generated by a speech recognition system?

> A spellchecker points to spelling errors and possibly suggests alternatives. An autocorrector usually goes a step further and automatically picks the most likely word. In case of the correct word already having been typed, the same is retained. So, in practice, an autocorrect is a bit more aggressive than a spellchecker, but this is more of an implementation detail — tools allow you to configure the behaviour.

- [language-models-spellchecking-and-autocorrection](https://towardsdatascience.com/language-models-spellchecking-and-autocorrection-dd10f739443c)
---

## How would you build a system to translate English text to Greek and vice-versa?

Use `seq2seq` learning model with `attention`


- [AnalyticsVidya](https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/)
- [TF Blog](https://www.tensorflow.org/tutorials/text/nmt_with_attention  )
----

## How would you build a system that automatically groups news articles by subject?

- Text Classification
- Topic Modelling


**Resource:**

- [Complete Guide to topic modelling with sci-kit learn and gensim](https://nlpforhackers.io/topic-modeling/)
- [Gensim topic modelling](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)
----

## How would you design a model to predict whether a movie review was positive or negative?

Typically, sentiment analysis for text data can be computed on several levels, including on an `individual sentence level`, `paragraph level`, or `the entire document` as a whole. Often, sentiment is computed on the document as a whole or some aggregations are done after computing the sentiment for individual sentences. There are two major approaches to sentiment analysis.

- Supervised machine learning or deep learning approaches
- Unsupervised lexicon-based approaches 

However most of the time we don't have the labelled data. So let's go for second approach. Hence, we will need to use unsupervised techniques for predicting the sentiment by using knowledgebases, `ontologies`, databases, and `lexicons` that have detailed information, specially curated and prepared just for sentiment analysis. 

Various popular lexicons are used for sentiment analysis, including the following.

1. AFINN lexicon
2. Bing Liu’s lexicon
3. MPQA subjectivity lexicon
4. SentiWordNet
5. VADER lexicon
6. TextBlob lexicon 

Use these lexicon, convert words to their sentiment

Actually there is no machine learning going on here but this library parses for every tokenized word, compares with its lexicon and returns the polarity scores. This brings up an overall sentiment score for the tweet.

- [Sentiment Analysis](https://www.kdnuggets.com/2018/08/emotion-sentiment-analysis-practitioners-guide-nlp-5.html)

### What is Lexicon and Ontologiy?

- A `lexicon` is a dictionary, vocabulary, or a book of words. In our case, lexicons are special dictionaries or vocabularies that have been created for analyzing sentiments.

- Ontologies provide semantic context. Identifying entities in unstructured text is a picture only half complete. Ontology models complete the picture by showing how these entities relate to other entities, whether in the document or in the wider world.

> An ontology is a formal and structural way of representing the concepts and relations of a shared conceptualization

![image](https://dw1.s81c.com/developerworks/mydeveloperworks/blogs/nlp/resource/nlp-shakespeare.jpg)

I realize that this sentence is really marked up and there’s arrows and red text going all over the place. So let’s examine this closely. 

- We’ve only recognized (e.g. `annotated`) two words in this entire sentence: William Shakespeare as a Playwright and Hamlet as a Play. But look at the depth of the understanding that we have. There’s a model depicted on this image, and we want to examine this more carefully. - 
- You’ll notice first of all that there are a total of 6 annotations represented on the diagram with arrows flowing between them. These annotations are produced by the NLP parser, and modeled (here’s the key point), they are modeled in the Ontology. It’s in the Ontology that we specify how a Book is related to a Date, or to a Language, and a Language to a Country to an Author, to a work produced by that Author, and so on.
- Each annotation is backed by a dictionary. The data for that dictionary is generated out of the triple store that conforms to the Ontology. The Ontology shows the relationship of all the annotations to each other.


## Why would someone want to develop an ontology? Some of the reasons are:

- To share common understanding of the structure of information among people or software agents
- To enable reuse of domain knowledge
- To make domain assumptions explicit
- To separate domain knowledge from the operational knowledge
- To analyze domain knowledge

![image](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/database/2018/10.1093_database_bay101/3/m_bay101f3.png?Expires=1574349961&Signature=e92Y-XDR47P6giB8FTMulFqrNjPKm4PSOGW~L6unhMGHWTdyAEBcY4BNMFAA1yfq4mgcBA~HtDnNGb3FGgQJwuJ35x2vNTE15I1t4zfZ88Nw5KVvteM7vH3310vyNhzeyVN9Gteh0TiwjENC6EtVjLzRpg73oz6jIy1RuCOlMurwhsqphFb3EjSiEd8jg9hydSDmZhxGVzDYyIC6LNxPGpnd66hcpI4BswbkoimwMXaWTFGrH~vWQk96UFMwT2Vmr9NqCHCoRkVI7CM5CIXuEqZdEAUEwWoMap3R~iM4YfkMvg7C-FBFrDlOEK0G-9vFfwfsLz~pDAXDc56ggZv0dw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

**Resource:**

- [Ontology Driven NLP](https://www.ibm.com/developerworks/community/blogs/nlp/entry/ontology_driven_nlp?lang=en)
- [Ontology](https://protege.stanford.edu/publications/ontology_development/ontology101-noy-mcguinness.html)
- [A survey of ontology learning technique and application](https://academic.oup.com/database/article/doi/10.1093/database/bay101/5116160)

## What is `Syntactic` analysis or parsing?

`Syntax analysis` or `syntactic analysis` is the process of analysing a string of symbols, either in natural language or in computer languages, conforming to the rules of a formal grammar. The term parsing comes from Latin pars (orationis), meaning part (of speech)

- Syntactic Analysis of a sentence is the task of recognising a sentence and assigning a syntactic structure to it. These syntactic structures are assigned by the Context Free Grammar (mostly PCFG) using parsing algorithms like Cocke-Kasami-Younger (CKY), Earley algorithm, Chart Parser. They are represented in a tree structure. These parse trees serve an important intermediate stage of representation for semantic analysis.

Syntactic Parse Tree 

![image](https://qphs.fs.quoracdn.net/main-qimg-979a3a252b5daeec31da78245623a450)

**Resource:**

- [Quora](https://www.quora.com/What-is-semantic-analysis-vs-syntactic-analysis-description-of-word-in-NLP)

### Concept of Parser

- It is used to implement the task of parsing. It may be defined as the software component designed for taking input data (text) and giving structural representation of the input after checking for correct syntax as per formal grammar. 
- It also builds a data structure generally in the form of parse tree or abstract syntax tree or other hierarchical structure

![image](https://www.tutorialspoint.com/natural_language_processing/images/symbol_table.jpg)


## What is `Semantic` analysis?

We already know that lexical analysis also deals with the meaning of the words, then how is semantic analysis different from lexical analysis? Lexical analysis is based on smaller token but on the other side semantic analysis focuses on larger chunks. That is why semantic analysis can be divided into the following two parts −

- The semantic analysis of natural language content starts by reading all of the words in content to capture the real meaning of any text. 
- It identifies the text elements and assigns them to their logical and grammatical role. 
- It analyzes context in the surrounding text and it analyzes the text structure to accurately disambiguate the proper meaning of words that have more than one definition.

- Semantic technology processes the logical structure of sentences to identify the most relevant elements in text and understand the topic discussed. 
- It also understands the relationships between different concepts in the text. 
  - For **example**, it understands that a text is about “politics” and “economics” even if it doesn’t contain the the actual words but related concepts such as “election,” “Democrat,” “speaker of the house,” or “budget,” “tax” or “inflation.”


Semantic analysis is a larger term, meaning to analyse the meaning contained within text, not just the sentiment. It looks for relationships among the words, how they are combined and how often certain words appear together.

**Resource:**

- [Semantic Analysis](https://expertsystem.com/natural-language-process-semantic-analysis-definition/)

## What id `Taxonomy`?



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
----
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


1. What are stop words? Describe an application in which stop words should be removed.
2.  How would you design a model to predict whether a movie review was positive or negative?
3.  Which is a better algorithm for POS tagging – SVM or hidden Markov models?
4.  What is the difference between shallow parsing and dependency parsing?
5.  What package are you aware of in python which is used in NLP and ML?
6.  Explain one application in which stop words should be removed.
7.  Which is better to use while extracting features character n-grams or word n-grams? Why?
8.  What is dimensionality reduction?
9.  Explain the working of SVM/NN/Maxent algorithms
10. Which is a better algorithm for POS tagging - SVM or hidden markov models ? why?
11. What packages are you aware of in python which are used in NLP and ML?
12. What are conditional random fields ?
13. When can you use Naive Bayes algorithm for training, what are its advantages and disadvantages?
14. How would you build a POS tagger from scratch given a corpus of annotated sentences? How would you deal with unknown words?


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