---
layout: post
title:  "Paper Summary - Word2Vec: Distributed Representation of Words (Part 1)"
date:   2020-01-24 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

----

# Introduction

We are going to summarize 2 papers 

- **Efficient Estimation of Word Representations in Vector Space** [[1]](#1)
  - Author: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
  - [Paper](https://arxiv.org/abs/1301.3781)
- **Distributed Representations of Words and Phrases and their Compositionality** [[2]](#2)
  - Author: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean 
  - [Paper](https://arxiv.org/abs/1310.4546)

----

- `Distributed representations of words` in a `vector space` help learning algorithms to achieve better performance in natural language processing tasks by `grouping similar words`. 
- Earlier there were label encoding, binarization or one-hot-encoding. Those were `sparse vector representation`. But the Skip-gram model helps to represent the words as `continuous dense vector`.
- The problem with label encoding, one-hot-encoding type vector representation is that, they don't capture the correlation (very loosely speaking) with each other. The correlation groups the words in terms of their hidden or latent meaning.

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Example

Let's say we have 4 words: `dog`, `cat`, `chair` and `table`.

If we apply one-hot-encoding then:

- dog: [0, 0]
- cat: [0, 1]
- chair: [1, 0]
- table: [1, 1]

The above is a random order. We can put in any random order that will not be a problem. Because all 4 words are not related in the vector space.


<center>
<img src="/assets/images/image_33_w2v_1.png" alt="image" height="300">
</center>

However `cat` and `dog` are from type _animal_ and `chair` and `table` are from type _furniture_. So it would have been very good if in the vector space they were grouped together and their vectors are adjusted accordingly.

<center>
<img src="/assets/images/image_33_w2v_2.png" alt="image" height="300">
</center>

Now there are many methods to learn these kind of representation. But at present 2 popular ones are `CBOW: Continuous Bag of Words` and `Skip-gram` model.

Both are kind of complement of each other. 

Say, we have a sentence $S = w_1 w_2 w_3 w_4 w_5$ where $w_i$ are words. Now say we pick word $w_3$ as our `candidate word` for which we are trying to get the dense vector. Then the remaining words are `neighbor words`. These neighbor words denote the context for the candidate words and the dense vector representation capture these properties.  

- Sentence $S = w_1 w_2 w_3 w_4 w_5$
- Candidate: $w_3$
- Neighbor:  $w_1 w_2 w_4 w_5$

# Objective 

The train objective is to learn word vector representation that are good at predicting the nearby words.

**CBOW Objective:** Predicts the candidate word $w_3$ based on neighbor words $w_1 w_2 w_4 w_5$. 


**Skip-gram Objective:** Predicts the Neighbor words $w_1 w_2 w_4 w_5$ based on candidate word $w_3$


![image](/assets/images/word2vec_3.png)

<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----
# Word2Vec

Let us now introduce arguably the most popular word embedding model, the model that launched a thousand word embedding papers: **word2vec**, the subject of two papers by Mikolov et al. in 2013. 

As word embeddings are a key building block of deep learning models for NLP, word2vec is often assumed to belong to the same group. Technically however, word2vec is not be considered to be part of deep learning, as its architecture is neither deep nor uses non-linearities (in contrast to Bengio's model and the C&W model).

In their first paper [[1]](#1), Mikolov et al. propose **two architectures** for learning word embeddings that are computationally less expensive than previous models. 

In their second paper [[2]](#2), they improve upon these models by employing additional strategies to enhance training speed and accuracy.

These architectures offer two main benefits over the C&W model [[4]](#4) and Bengio's language model [[3]](#3):

- They do away with the expensive hidden layer.
- They enable the language model to take additional context into account.

As we will later show, the success of their model is not only due to these changes, but especially due to certain training strategies.

**Side-note:** `word2vec` and `GloVe` might be said to be to NLP what VGGNet is to vision, i.e. a common weight initialization that provides generally helpful features without the need for lengthy training.



In the following, we will look at both of these architectures:

## Continuous bag-of-words (CBOW)

Mikolov et al. thus use both the $n$ words before and after the target word $w_t$ to predict it. They call this continuous bag-of-words (CBOW), as it uses continuous representations whose order is of no importance.

The objective function of CBOW in turn is only slightly different than the language model one:


$$
J_\theta = \frac{1}{T}\sum\limits_{t=1}^T\ \log p(w_t \vert w_{t-n}^{t-1},w_{t+1}^{t+n})
$$

where $w_{t-n}^{t-1}=w_{t-n} , \cdots , w_{t-1}$ and $w_{t+1}^{t+n}=w_{t+1}, \cdots , w_{t+n}$ 

## The Skip-gram model

While CBOW can be seen as a precognitive language model, skip-gram turns the language model objective on its head: Instead of using the surrounding words to predict the centre word as with CBOW, skip-gram uses the centre word to predict the surrounding words

The training objective of the Skip-gram model is to find word representations that are useful for predicting the surrounding words in a sentence or a document. More formally, given a sequence of training words $w_1, w_2, w_3, \dots , w_T$, the objective of the Skip-gram model is to maximize the average log probability, i.e

$$J_\theta = \frac{1}{T} \sum\limits_{t=1}^T \sum\limits_{-c \leq j \leq c , j \ne 0} \log{p(w_{t+j}\vert w_t)}$$

where $c$ is the size of the training context (which can be a function of the center word $w_t$). Larger $c$ results in more training examples and thus can lead to a higher accuracy, at the expense of the training time.

The basic Skip-gram formulation defines $p(w_{t+j} \vert w_t)$ using the softmax function. 

- A computationally `efficient approximation` of the full softmax is the **hierarchical softmax**. The main advantage is that instead of evaluating $W$ output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about $\log_2(W)$ nodes.
- The hierarchical softmax uses a binary tree representation of the output layer with the W words as its leaves and, for each node, explicitly represents the relative probabilities of its child nodes. These define a random walk that assigns probabilities to words.
- Such representation makes the learning faster using distributed technique.

----

## GloVe: Global Vectors for Word Representation

GloVe [[5]](#5) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

The authors of Glove show that the ratio of the co-occurrence probabilities of two words (rather than their co-occurrence probabilities themselves) is what contains information and aim to encode this information as vector differences.

To achieve this, they propose a weighted least squares objective $J$ that directly aims to minimize the difference between the dot product of the vectors of two words and the logarithm of their number of co-occurrences:

$$
J = \sum\limits_{i, j=1}^V f(X_{ij})   (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \text{log}   X_{ij})^2
$$


where $w_i$ and $b_i$ are the word vector and bias respectively of word $i$, $\tilde{w_j}$ and $b_j$ are the context word vector and bias respectively of word $j$, $X_{ij}$ is the number of times word $i$ occurs in the context of word $j$, and $f$ is a weighting function that assigns relatively lower weight to rare and frequent co-occurrences.


**Reference:**

- <a id="1">[1]</a> 
Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. (ICLR 2013)
- <a id="2">[2]</a> 
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. NIPS
- <a id="3">[3]</a> 
Bengio, Y., Ducharme, R., Vincent, P., & Janvin, C. (2003). A Neural Probabilistic Language Model. The Journal of Machine Learning Research
- <a id="4">[4]</a> 
Collobert, R., & Weston, J. (2008). A unified architecture for natural language processing. ICML ’08,
- <a id="5">[5]</a> 
Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global Vectors for Word Representation, EMNLP 2014


- [On word embeddings - Part 1: Sebastian Ruder](https://ruder.io/word-embeddings-1/index.html)
- [Language Model: CS124](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>