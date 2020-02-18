---
layout: post
title:  "Deep Learning: Natural Language Processing (Part 2)"
date:   2020-02-18 00:00:10 -0030
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}
---

Quick Refresher: Neural Network in NLP


# Sentence Transformers: Sentence Embeddings using BERT a.k.a Sentence BERT

From the abstract of the original paper

**BERT** (Devlin et al., $2018$) and **RoBERTa** (Liu et al., 2019) has set a new state-of-the-art performance on **sentence-pair regression** tasks like `semantic textual similarity` (STS). However, it requires that both sentences are fed into the network, which causes a massive computational overhead: Finding the most similar pair in a collection of 10,000 sentences requires about 50 million inference computations (~65 hours) with BERT. The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering.

In this paper, we present **Sentence-BERT** (SBERT), a modification of the pretrained BERT network that use `siamese` and `triplet network` structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT. 

![image](/assets/images/image_06_SBERT_1.png)


## How to use it in code

```py
# install the package
pip install -U sentence-transformers
```

```py
# download a pretrained model.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Then provide some sentences to the model.
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']
sentence_embeddings = model.encode(sentences)

# And that's it already. We now have a list of numpy arrays with the embeddings.
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```

_**for more details check the pypi repository_


**Reference:**

- [PyPi sentence-transformers](https://pypi.org/project/sentence-transformers/#Training)
- [arXiv: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)


<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# What is Siamese Network

A **twin neural network** (sometimes called a Siamese Network, though this term is frowned upon) is an artificial neural network that **uses the same weights** while working in tandem (having two things arranged one in front of the other) on two different input vectors to compute comparable output vectors. Often one of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared. This is similar to comparing fingerprints but can be described more technically as a distance function for `locality-sensitive hashing`.

It is possible to make a kind of structure that is functional similar to a siamese network, but implements a slightly different function. This is typically used for comparing similar instances in different type sets.

Uses of similarity measures where a twin network might be used are such things as 

- Recognizing handwritten checks
- Automatic detection of faces in camera images
- Matching queries with indexed documents.

## Learning

Learning in twin networks can be done with `triplet loss` or `contrastive loss`.

### Triplet Loss

Triplet loss is a loss function for artificial neural networks where a baseline (`anchor`) input is compared to a positive (`truthy`) input and a negative (`falsy`) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized. [wiki](https://en.wikipedia.org/wiki/Triplet_loss)

- minimize distance(baseline,truth)
- maximize distance(baseline,false) 

The negative (false) vector will force learning in the network, while the positive vector (truth) will act like a regularizer.

### Predefined metrics, Euclidean distance metric

The common learning goal is to minimize a distance metric for similar objects and maximize for distinct ones. This gives a loss function like 

$$
\delta(x^i, x^j)=\left\{
                \begin{array}{ll}
                  min \vert\vert f(x^i) - f(x^j) \vert\vert, i \ne j\\
                  max \vert\vert f(x^i) - f(x^j) \vert\vert, i = j
                \end{array}
              \right.
$$


### Twin Networks for Object Tracking

Twin networks have been used in object tracking because of its unique two tandem inputs and **similarity measurement**. In object tracking, one input of the twin network is user pre-selected exemplar image, the other input is a larger search image, which twin network's job is to locate exemplar inside of search image. By measuring the similarity between exemplar and each part of the search image, a map of similarity score can be given by the twin network. 

Furthermore, using a Fully Convolutional Network, the process of computing each sector's similarity score can be replaced with only one cross correlation layer.


**Reference:**

- [wiki: Siamese network](https://en.wikipedia.org/wiki/Siamese_neural_network)
- [wiki: triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)

----

# Exercise:

- U-Net
- UMLFit

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>