---
layout: post
title:  "Blog 600: Paper Summary - Distributed Representation of Words"
date:   2019-07-28 00:11:31 +0530
categories: jekyll update
mathjax: true
---

# Content

1. TOC
{:toc}

----


# Big picture

- `Distributed representations of words` in a `vector space` help learning algorithms to achieve better performance in natural language processing tasks by `grouping similar words`. 
- Earlier there were label encoding, binarization or one-hot-encoding. Those were `sparse vector representation`. But the Skip-gram model helps to represent the words as `continuous dense vector`.
- The problem with label encoding, one-hot-encoding type vector representation is that, they don't capture the correlation (very loosely speaking) with each other. The correlation groups the words in terms of their hidden or latent meaning.

# Example

Let's say we have 4 words: `dog`, `cat`, `chair` and `table`.

If we apply one-hot-encoding then:

- dog: [0, 0]
- cat: [0, 1]
- chair: [1, 0]
- table: [1, 1]

The above is a random order. We can put in any random order that will not be a problem. Because all 4 words are not related in the vector space.


![image](/assets/images/word2vec.svg)

However `cat` and `dog` are from type _animal_ and `chair` and `table` are from type _furniture_. So it would have been very good if in the vector space they were grouped together and their vectors are adjusted accordingly.


![image](/assets/images/word2vec_2.svg)



  - Skip-gram model does not involve dense matrix multiplications. This makes the training extremely efficient.

# Objective Skip-gram

The train objective is to learn word vector representation that are good at predicting the nearby words.

## Why it's called Skip-gram?

If we remember, `N-Gram` _means a sequence of N words_. However in Skip-gram we are kind of skipping or masking the word and our objective is given the masked word predict it's surrounding words. From there the name comes skip-gram.