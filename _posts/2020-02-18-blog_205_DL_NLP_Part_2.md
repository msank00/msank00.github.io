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



<a href="#Top" style="color:#2F4F4F;background-color: #c8f7e4;float: right;">Content</a>

----

# Snorkel: Rapid Training Data Creation with Weak Supervision


## What is Weak Supervision

According to this [bolg](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)

>  Noisier or higher-level supervision is used as a more expedient and flexible way to get supervision signal, in particular from subject matter experts (SMEs).

![image](https://www.snorkel.org/doks-theme//assets/images/2017-07-16-weak-supervision/WS_mapping.png)


- In **active learning**, the goal is to make use of subject matter experts more efficiently by having them **label data points which are estimated to be most valuable to the model**. For example, we might select mammograms that lie close to the current model decision boundary, and ask radiologists to label only these. 
- In the **semi-supervised learning** setting, we have a small labeled training set and a much larger unlabeled data set. At a high level, we then use **assumptions about the structure of the data** like `smoothness`, `low dimensional structure`, or `distance metrics` to leverage the unlabeled data (either as part of a generative model, as a regularizer for a discriminative model, or to learn a compact data representation). Broadly, rather than soliciting more input from subject matter experts, the idea in semi-supervised learning is to leverage domain- and task-agnostic assumptions to exploit the unlabeled data that is often cheaply available in large quantities.
- In the standard **transfer learning** setting, our goal is to take one or more models already trained on a different dataset and apply them to our dataset and task. For example, we might have a large training set for tumors in another part of the body, and classifiers trained on this set, and wish to apply these somehow to our mammography task.

_The above paradigms potentially allow us to avoid asking our SME collaborators for additional training labels._

But what if–either in addition, or instead–we could ask SME for various types of higher-level, or otherwise less precise, forms of supervision, which would be faster and easier to provide? For example, what if our radiologists could spend an afternoon specifying a **set of heuristics** or other resources, that–if handled properly–could effectively replace thousands of training labels? This is the key practical motivation for weak supervision approaches,

:rocket: **Heuristic Examples**

```r
# Return a label of SPAM if "http" link in email text, otherwise ABSTAIN

# Return a label of SPAM if substring like "my channel", "my video" are there in the email text. 
```

<center>
<img src="https://www.snorkel.org/doks-theme//assets/images/2017-07-16-weak-supervision/WS_diagram.png" width="600">
</center>


## Data Programming:

Data programming: A paradigm for the `programmatic creation` and `modeling of training datasets`. Data programming provides a simple, unifying framework for weak supervision, in which training labels are noisy and may be from multiple, potentially overlapping sources.

In data programming, users encode this weak supervision in the form of labeling functions, which are user-defined programs that each provide a label for some subset of the data, and collectively generate a large but potentially overlapping set of training examples. 

Many different weak supervision approaches can be expressed as labeling functions.

However `labeling functions` may have widely varying error rates and may conflict on certain data points. To address this, we model the labeling functions as a **generative process**, which lets us automatically de-noise the resulting training set by learning the accuracies of the labeling functions along with their correlation structure.

Think data programming as a paradigm by modeling multiple label sources without access to ground truth, and **generating probabilistic training labels** representing the lineage of the individual labels. 

## Snorkel architecture

From the original Snorkle [paper](https://link.springer.com/article/10.1007/s00778-019-00552-1), the Snorkel architecture is as follows: 

**Writing Labeling Functions:** Rather than hand labeling training data, users of Snorkel write labeling functions, which allow them to express various weak supervision sources such as patterns, heuristics, external knowledge bases, and more. 

**Modeling Accuracies and Correlations:** Snorkel automatically learns a generative model over the labeling functions, which allows it to estimate their `accuracies` and `correlations`. This step uses no ground-truth data, **learning instead from the agreements and disagreements of the labeling functions**.


**Training a Discriminative Model:** The output of Snorkel is a set of `probabilistic labels` that can be used to train a wide variety of state-of-the-art machine learning models, such as popular deep learning models. While the generative model is essentially a **re-weighted combination of the user-provided labeling functions**, which tend to be precise but low-coverage.

**Reference:**

- [arXiv Paper](https://arxiv.org/abs/1711.10160)
- [Snorkel Resources](https://www.snorkel.org/resources/)
- [Weak Supervision: The New Programming Paradigm for Machine Learning](http://ai.stanford.edu/blog/weak-supervision/)
- [Book: Semi Supervised Learning](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf)

**_to be continued_**

----

# Exercise:

- UMLFit
- U-Net

----


<a href="#Top" style="color:#023628;background-color: #f7d06a;float: right;">Back to Top</a>