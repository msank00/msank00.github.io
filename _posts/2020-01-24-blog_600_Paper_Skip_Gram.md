---
layout: post
title:  "Paper Summary - Distributed Representation of Words (Part 1)"
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

- **Efficient Estimation of Word Representations in Vector Space**
  - Author: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
  - [Link](https://arxiv.org/abs/1301.3781)
- **Distributed Representations of Words and Phrases and their Compositionality**
  - Author: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean 
  - [Link](https://arxiv.org/abs/1310.4546)

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

# The Skip-gram model

The training objective of the Skip-gram model is to find word representations that are useful for predicting the surrounding words in a sentence or a document. More formally, given a sequence of training words $w_1, w_2, w_3, \dots , w_T$, the objective of the Skip-gram model is to maximize the average log probability, i.e

$$\frac{1}{T} \Sigma_{t=1}^T \Sigma_{-c \leq j \leq c , j \ne 0} \log{p(w_{t+j}\vert w_t)}$$

where $c$ is the size of the training context (which can be a function of the center word $w_t$). Larger $c$ results in more training examples and thus can lead to a higher accuracy, at the expense of the training time.

The basic Skip-gram formulation defines $p(w_{t+j} \vert w_t)$ using the softmax function. 

- A computationally `efficient approximation` of the full softmax is the **hierarchical softmax**. The main advantage is that instead of evaluating $W$ output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about $\log_2(W)$ nodes.
- The hierarchical softmax uses a binary tree representation of the output layer with the W words as its leaves and, for each node, explicitly represents the relative probabilities of its child nodes. These define a random walk that assigns probabilities to words.
- Such representation makes the learning faster using distributed technique.

**Resource:**

- [On word embeddings - Part 1: Sebastian Ruder](https://ruder.io/word-embeddings-1/index.html)
- [Language Model: CS124](https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf)

----

<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>